#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/time.h>
/**
	Timing functions
*/
typedef long unsigned int timestamp_t;

inline timestamp_t current_time_usecs() __attribute__((always_inline));
inline timestamp_t current_time_usecs(){
	struct timeval t;
	gettimeofday(&t, NULL);
	return (t.tv_sec)*1000000L + t.tv_usec;

}

inline long current_time_nsecs() __attribute__((always_inline));
inline long current_time_nsecs(){
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (t.tv_sec)*1000000000L + t.tv_nsec;
}
#endif // UTILS_HPP
