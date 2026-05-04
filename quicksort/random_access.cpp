#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <numeric>
#include <random>
#include <thread>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>

constexpr int NUM_THREADS = 1;
constexpr size_t DEFAULT_FRAGMENT_OCCUPIED_PERCENT = 75;
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

double to_mb(size_t bytes) {
	return static_cast<double>(bytes) / MB;
}

size_t parse_fragment_occupied_percent(int argc, char *argv[]) {
	const char *env_value = std::getenv("RANDOM_ACCESS_FRAGMENT_PERCENT");
	const char *raw_value = nullptr;

	if (argc >= 3)
		raw_value = argv[2];
	else if (env_value != nullptr)
		raw_value = env_value;
	else
		return DEFAULT_FRAGMENT_OCCUPIED_PERCENT;

	char *end = nullptr;
	errno = 0;
	const unsigned long parsed = std::strtoul(raw_value, &end, 10);
	if (errno != 0 || end == raw_value || *end != '\0')
		die("fragment occupied percent must be an integer", false);
	if (parsed == 0 || parsed >= 100)
		die("fragment occupied percent must be between 1 and 99", false);

	return static_cast<size_t>(parsed);
}

size_t release_random_holes(char *arena,
	size_t total_pages,
	size_t release_pages,
	size_t page_size) {
	const size_t chunk_pages = std::max<size_t>(1, FRAGMENT_RELEASE_CHUNK_BYTES / page_size);
	const size_t total_chunks = (total_pages + chunk_pages - 1) / chunk_pages;
	const size_t full_release_chunks = release_pages / chunk_pages;
	const size_t tail_release_pages = release_pages % chunk_pages;

	std::vector<size_t> chunk_ids(total_chunks);
	std::iota(chunk_ids.begin(), chunk_ids.end(), 0);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	std::shuffle(chunk_ids.begin(), chunk_ids.end(), gen);

	std::vector<size_t> full_chunk_ids(chunk_ids.begin(), chunk_ids.begin() + full_release_chunks);
	std::sort(full_chunk_ids.begin(), full_chunk_ids.end());

#ifndef MADV_DONTNEED
	die("MADV_DONTNEED is unavailable on this platform", false);
#endif

	size_t hole_count = 0;
	size_t released_pages = 0;
	if (!full_chunk_ids.empty()) {
		size_t run_start_chunk = full_chunk_ids.front();
		size_t run_end_chunk = run_start_chunk + 1;

		for (size_t i = 1; i <= full_chunk_ids.size(); ++i) {
			const bool contiguous = i < full_chunk_ids.size() && full_chunk_ids[i] == run_end_chunk;
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

			if (i < full_chunk_ids.size()) {
				run_start_chunk = full_chunk_ids[i];
				run_end_chunk = run_start_chunk + 1;
			}
		}
	}

	if (tail_release_pages != 0) {
		size_t tail_chunk = 0;
		bool found_tail_chunk = false;
		for (size_t i = full_release_chunks; i < chunk_ids.size(); ++i) {
			const size_t candidate_chunk = chunk_ids[i];
			const size_t chunk_start_page = candidate_chunk * chunk_pages;
			const size_t chunk_end_page = std::min(total_pages, chunk_start_page + chunk_pages);
			if (chunk_end_page - chunk_start_page >= tail_release_pages) {
				tail_chunk = candidate_chunk;
				found_tail_chunk = true;
				break;
			}
		}
		if (!found_tail_chunk)
			die("failed to place random tail hole", false);

		const size_t tail_chunk_start_page = tail_chunk * chunk_pages;
		const size_t tail_chunk_end_page = std::min(total_pages, tail_chunk_start_page + chunk_pages);
		const size_t tail_chunk_capacity = tail_chunk_end_page - tail_chunk_start_page;
		std::uniform_int_distribution<size_t> tail_offset_dis(0,
			tail_chunk_capacity - tail_release_pages);
		const size_t tail_start_page =
			tail_chunk_start_page + tail_offset_dis(gen);
		if (madvise(arena + tail_start_page * page_size,
			tail_release_pages * page_size,
			MADV_DONTNEED) != 0)
			die("MADV_DONTNEED failed for random tail hole", true);

		released_pages += tail_release_pages;
		++hole_count;
	}

	std::cout << "released " << std::fixed << std::setprecision(2)
		<< to_mb(released_pages * page_size)
		<< " MB in " << hole_count
		<< " random holes (chunk size " << FRAGMENT_RELEASE_CHUNK_BYTES / MB
		<< " MB)\n" << std::defaultfloat;

	return released_pages;
}

size_t pre_fragment_swap_entries(size_t total_fragment_size, size_t occupied_percent) {
	const long page_size = sysconf(_SC_PAGESIZE);
	if (page_size <= 0)
		die("failed to get page size", true);

	const size_t aligned_total_fragment_size =
		align_up(total_fragment_size, static_cast<size_t>(page_size));
	const size_t total_fragment_pages =
		aligned_total_fragment_size / static_cast<size_t>(page_size);
	const size_t free_percent = 100 - occupied_percent;
	size_t release_pages = (total_fragment_pages * free_percent + 99) / 100;
	release_pages = std::max<size_t>(1, release_pages);
	if (release_pages >= total_fragment_pages)
		die("total fragmentation size is too small for the requested occupied percent", false);

	const size_t fragmented_space = aligned_total_fragment_size;

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

	std::cout << "allocated " << std::fixed << std::setprecision(2)
		<< to_mb(fragmented_space)
		<< " MB for pre-fragmentation before releasing random holes"
		<< " (target retained ratio " << occupied_percent << "%)\n"
		<< std::defaultfloat;

#ifdef MADV_PAGEOUT
	if (madvise(fragmented_arena, fragmented_space, MADV_PAGEOUT) != 0)
		die("MADV_PAGEOUT failed for fragmentation arena", true);
	std::cout << "requested MADV_PAGEOUT for all pre-fragmentation pages\n";
#else
	std::cout << "MADV_PAGEOUT is unavailable; random hole release will not pre-occupy swap eagerly\n";
#endif

	const size_t actual_release_pages = release_random_holes(fragmented_arena,
		total_fragment_pages,
		release_pages,
		static_cast<size_t>(page_size));
	const size_t actual_occupied_pages = total_fragment_pages - actual_release_pages;
	const double actual_occupied_percent =
		100.0 * static_cast<double>(actual_occupied_pages) / total_fragment_pages;
	const size_t active_size = actual_release_pages * static_cast<size_t>(page_size);
	std::cout << "retained " << std::fixed << std::setprecision(2)
		<< to_mb(actual_occupied_pages * static_cast<size_t>(page_size))
		<< " MB of fragmented occupancy and freed "
		<< to_mb(active_size) << " MB for later random access"
		<< " (actual retained ratio " << actual_occupied_percent << "%)\n"
		<< std::defaultfloat;

	return active_size;
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
	if (argc < 2 || argc > 3)
		die("usage: random_access <total_fragment_mb> [fragment_occupied_percent]", false);
	const size_t total_fragment_mb = std::stoull(argv[1]);
	if (total_fragment_mb == 0)
		die("total_fragment_mb must be greater than 0", false);
	const size_t occupied_percent = parse_fragment_occupied_percent(argc, argv);
	const size_t fragment_total_size = total_fragment_mb * MB;
	const size_t size = pre_fragment_swap_entries(fragment_total_size, occupied_percent);
	const size_t numInts = size / sizeof(std::atomic<int>);

	std::cout << "will random access " << numInts << " integers ("
		<< std::fixed << std::setprecision(2) << to_mb(size)
		<< " MB) after reserving " << occupied_percent
		<< "% of the total space\n" << std::defaultfloat;
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
