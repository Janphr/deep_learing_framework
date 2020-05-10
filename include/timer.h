//
// Created by janphr on 18.02.20.
//

#ifndef VIS_LOC_TIMER_H
#define VIS_LOC_TIMER_H

#include <chrono>

class Timer{

public:
    static constexpr double SECONDS = 1e-9;
    static constexpr double MILLISECONDS = 1e-6;
    static constexpr double NANOSECONDS = 1.0;

    explicit Timer(double scale) : scale(scale) {}
    Timer() {}

    void start() {
        started = true;
        start_t = chrono::high_resolution_clock::now();
    }

    double time() {
        end_t = chrono::high_resolution_clock::now();
        chrono::duration<double, nano> elapsed_ns = end_t - start_t;
        return elapsed_ns.count()*scale;
    }

    double stop() {

        if (!started)
            throw std::logic_error("[Timer] Stop called without previous start");

        started = false;
        return time();
    }



private:
    chrono::high_resolution_clock::time_point start_t;
    chrono::high_resolution_clock::time_point end_t;
    bool started;
    double scale = MILLISECONDS;
};



#endif //VIS_LOC_TIMER_H
