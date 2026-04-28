#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <atomic>
#include <numeric>
#include <random>
#include <thread>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>

constexpr int NUM_THREADS = 32;
constexpr size_t FRAGMENT_OCCUPIED_PARTS = 3;
constexpr size_t FRAGMENT_TOTAL_PARTS = 4;
constexpr size_t FRAGMENT_RELEASE_CHUNK_BYTES = 1 * 1024 * 1024;

const size_t MB = 1024 * 1024;
static std::vector<void *> g_fragmentation_arenas;

using namespace std::chrono;

void die(const char *msg, bool printErrno) {
	std::cerr << msg;
	if (printErrno)
		std::cerr << ": " << std::strerror(errno);
	std::cerr << "\n";
	exit(1);
}

void print_time_diff(time_point<high_resolution_clock> start,
	time_point<high_resolution_clock> end) {
	auto diff = end - start;
	std::cout << "time " << duration<double, std::nano> (diff).count() << "\n";
}

void print_time_diff_ms(time_point<high_resolution_clock> start,
	time_point<high_resolution_clock> end) {
	auto diff = end - start;
	std::cout << "time " << duration<double, std::milli> (diff).count() << " ms\n";
}

size_t align_up(size_t value, size_t alignment) {
	return ((value + alignment - 1) / alignment) * alignment;
}

void release_random_holes(char *arena,
	size_t total_pages,
	size_t release_pages,
	size_t page_size) {
	const size_t chunk_pages = std::max<size_t>(1, FRAGMENT_RELEASE_CHUNK_BYTES / page_size);
	const size_t total_chunks = (total_pages + chunk_pages - 1) / chunk_pages;
	const size_t release_chunks = (release_pages + chunk_pages - 1) / chunk_pages;

	std::vector<size_t> chunk_ids(total_chunks);
	std::iota(chunk_ids.begin(), chunk_ids.end(), 0);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	std::shuffle(chunk_ids.begin(), chunk_ids.end(), gen);
	chunk_ids.resize(release_chunks);
	std::sort(chunk_ids.begin(), chunk_ids.end());

#ifndef MADV_DONTNEED
	die("MADV_DONTNEED is unavailable on this platform", false);
#endif

	size_t hole_count = 0;
	size_t released_pages = 0;
	if (!chunk_ids.empty()) {
		size_t run_start_chunk = chunk_ids.front();
		size_t run_end_chunk = run_start_chunk + 1;

		for (size_t i = 1; i <= chunk_ids.size(); ++i) {
			const bool contiguous = i < chunk_ids.size() && chunk_ids[i] == run_end_chunk;
			if (contiguous) {
				++run_end_chunk;
				continue;
			}

			const size_t start_page = run_start_chunk * chunk_pages;
			const size_t end_page = std::min(total_pages, run_end_chunk * chunk_pages);
			const size_t run_pages = end_page - start_page;
			if (madvise(arena + start_page * page_size, run_pages * page_size, MADV_DONTNEED) != 0)
				die("MADV_DONTNEED failed for random hole", true);

			released_pages += run_pages;
			++hole_count;

			if (i < chunk_ids.size()) {
				run_start_chunk = chunk_ids[i];
				run_end_chunk = run_start_chunk + 1;
			}
		}
	}

	std::cout << "released " << (released_pages * page_size) / MB
		<< " MB in " << hole_count
		<< " random holes (chunk size " << FRAGMENT_RELEASE_CHUNK_BYTES / MB
		<< " MB)\n";
}

void pre_fragment_swap_entries(size_t active_size) {
	const long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0)
		die("failed to get page size", true);

	const size_t aligned_active_size = align_up(active_size, static_cast<size_t>(page_size));
	const size_t active_pages = aligned_active_size / static_cast<size_t>(page_size);
	const size_t total_fragment_pages = active_pages * FRAGMENT_TOTAL_PARTS;
	const size_t release_pages = active_pages;
	const size_t occupied_pages = active_pages * FRAGMENT_OCCUPIED_PARTS;
	const size_t fragmented_space = total_fragment_pages * static_cast<size_t>(page_size);

	char *fragmented_arena = static_cast<char *>(mmap(nullptr,
		fragmented_space,
		PROT_READ | PROT_WRITE,
		MAP_ANONYMOUS | MAP_PRIVATE,
		-1,
		0));
	if (fragmented_arena == MAP_FAILED)
		die("fragmentation mmap failed", true);
	// Keep the mapping alive so the swap entries stay occupied for the workload.
	g_fragmentation_arenas.push_back(fragmented_arena);

	for (size_t page = 0; page < total_fragment_pages; ++page)
		fragmented_arena[page * page_size] = static_cast<char>(page);

	std::cout << "allocated " << fragmented_space / MB
		<< " MB for pre-fragmentation before releasing random holes\n";

#ifdef MADV_PAGEOUT
	if (madvise(fragmented_arena, fragmented_space, MADV_PAGEOUT) != 0)
		die("MADV_PAGEOUT failed for fragmentation arena", true);
	std::cout << "requested MADV_PAGEOUT for all pre-fragmentation pages\n";
#else
	std::cout << "MADV_PAGEOUT is unavailable; random hole release will not pre-occupy swap eagerly\n";
#endif

	release_random_holes(fragmented_arena,
		total_fragment_pages,
		release_pages,
		static_cast<size_t>(page_size));
	std::cout << "retained " << (occupied_pages * static_cast<size_t>(page_size)) / MB
		<< " MB of fragmented occupancy for later random access\n";
}

void random_add(size_t numInts) {
	auto atomic_vector = new std::vector<std::atomic<int>>(numInts);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	std::uniform_int_distribution<size_t> dis(0, atomic_vector->size() - 1);

	while(true) {
		size_t idx = dis(gen);
		atomic_vector->at(idx).fetch_add(1);
	}
}

int main(int argc, char *argv[]) {
	if (argc != 2)
		die("need MB of random-access working set", false);
	const size_t size = std::stoull(argv[1]) * MB;
	const size_t numInts = size / sizeof(std::atomic<int>);

	pre_fragment_swap_entries(size);

	std::cout << "will random access " << numInts << " integers (" << size / MB << " MB)\n";
	std::vector<std::thread> threads;
	threads.reserve(NUM_THREADS);

	for(int i = 0; i < NUM_THREADS; ++i) {
		threads.emplace_back(random_add, numInts / NUM_THREADS);
	}

	for(auto &thread : threads) {
		thread.join();
	}

	return 0;
}
