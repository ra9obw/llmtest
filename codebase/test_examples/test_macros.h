// test_macros.h
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define DEBUG_MODE 1

#ifdef DEBUG_MODE
    #define LOG(msg) std::cerr << msg << std::endl
#else
    #define LOG(msg)
#endif