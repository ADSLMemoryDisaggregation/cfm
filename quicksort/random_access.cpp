#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <random>
#include <thread>
#include <sys/mman.h>

#define NUM_THREADS 32

const size_t MB = 1024 * 1024;
using namespace std::chrono;

void die(const char *msg, bool printErrno) {
	std::cerr << msg << "\n";
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

void random_add(long numInts) {
	auto atomic_vector = new std::vector<std::atomic<int>>(numInts);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);  
    std::uniform_int_distribution<> dis(0, atomic_vector->size() - 1); 
	int i = 0;
	uint64_t count = 0;
	uint64_t mid = atomic_vector->size() / 2;
	
    
    while(true) {
		uint64_t idx = dis(gen);
		
		atomic_vector->at(idx).fetch_add(1);
		
		count++;
		//std::this_thread::sleep_for(std::chrono::microseconds(2));
    }

}

int main(int argc, char *argv[]) {
	if (argc != 2)
		die("need MB of integers to sort", false);
    long raw_size = std::stoi(argv[1]) * MB;
	void* fragment[raw_size / MB];
    long size = raw_size / 4;
	long numInts = size / sizeof(std::atomic<int>);

    for(int i = 0; i < raw_size / MB; ++i) {
        fragment[i] = mmap(NULL, MB, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();  
    std::mt19937 gen(seed);  
    std::uniform_int_distribution<> dis(0, raw_size / MB - 1);
    for(int i = 0; i < size / MB; ++i) {
        int idx = dis(gen);
        munmap(fragment[idx], MB);
    }

	std::cout << "will random access " << numInts << " integers (" << size / MB << " MB)\n";
	//std::vector<std::atomic<int>> v(numInts);

	std::srand(std::time(0));
	time_point<high_resolution_clock> start, end;

	//std::generate(v.begin(), v.end(), std::rand);
	//start = high_resolution_clock::now();

	std::thread* threads[NUM_THREADS];
	for(int i = 0; i < NUM_THREADS; ++i) {
		threads[i] = new std::thread(random_add, (numInts/NUM_THREADS));
	}

	for(int i = 0; i < NUM_THREADS; ++i) {
		threads[i]->join();
	}

	//end = high_resolution_clock::now();
	//print_time_diff_ms(start, end);

	return 0;
}
