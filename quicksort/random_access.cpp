#include <iostream>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <atomic>
#include <random>
#include <thread>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>

constexpr int NUM_THREADS = 32;
constexpr size_t FRAGMENT_OCCUPIED_PARTS = 3;
constexpr size_t FRAGMENT_TOTAL_PARTS = 4;

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

void pre_fragment_swap_entries(size_t active_size) {
	const long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0)
		die("failed to get page size", true);

	const size_t aligned_active_size = align_up(active_size, static_cast<size_t>(page_size));
	const size_t active_pages = aligned_active_size / static_cast<size_t>(page_size);
	const size_t total_fragment_pages = active_pages * FRAGMENT_TOTAL_PARTS;
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

	for (size_t page = 0; page < total_fragment_pages; ++page) {
		const bool leave_hole =
			((page + 1) * active_pages / total_fragment_pages) !=
			(page * active_pages / total_fragment_pages);
		if (leave_hole)
			continue;
		fragmented_arena[page * page_size] = static_cast<char>(page);
	}

	std::cout << "pre-fragmented swap-entry space: total "
		<< fragmented_space / MB << " MB, occupied "
		<< (occupied_pages * static_cast<size_t>(page_size)) / MB << " MB, free "
		<< aligned_active_size / MB << " MB\n";

#ifdef MADV_PAGEOUT
	if (madvise(fragmented_arena, fragmented_space, MADV_PAGEOUT) != 0)
		die("MADV_PAGEOUT failed for fragmentation arena", true);
	std::cout << "requested MADV_PAGEOUT for pre-fragmented pages\n";
#else
	std::cout << "MADV_PAGEOUT is unavailable; fragmentation pages stay resident until reclaim\n";
#endif
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
