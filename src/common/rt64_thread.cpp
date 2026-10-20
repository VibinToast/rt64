//
// RT64
//

#include "rt64_thread.h"

#include <cassert>
#include <thread>

#if defined(_WIN64)
#   include <Windows.h>
#   include "utf8conv/utf8conv.h"
#elif defined(__linux__)
#   include <pthread.h>
#   include <sys/resource.h>
#   include <sys/syscall.h>
#   include <unistd.h>
#endif

namespace RT64 {
#   if defined(_WIN32)
    static int toWindowsPriority(Thread::Priority priority) {
        switch (priority) {
        case Thread::Priority::Idle:
            return THREAD_PRIORITY_IDLE;
        case Thread::Priority::Lowest:
            return THREAD_PRIORITY_LOWEST;
        case Thread::Priority::Low:
            return THREAD_PRIORITY_BELOW_NORMAL;
        case Thread::Priority::Normal:
            return THREAD_PRIORITY_NORMAL;
        case Thread::Priority::High:
            return THREAD_PRIORITY_ABOVE_NORMAL;
        case Thread::Priority::Highest:
            return THREAD_PRIORITY_HIGHEST;
        default:
            assert(false && "Unknown thread priority.");
            return THREAD_PRIORITY_NORMAL;
        }
    }
#   endif

    // Thread

    void Thread::setCurrentThreadName(const std::string &str) {
#   if defined(_WIN32)
        std::wstring nameWide = win32::Utf8ToUtf16(str);
        SetThreadDescription(GetCurrentThread(), nameWide.c_str());
#   elif defined(__linux__)
        pthread_setname_np(pthread_self(), str.c_str());
#   elif defined(__APPLE__)
        pthread_setname_np(str.c_str());
#   else
        static_assert(false, "Unimplemented");
#   endif
    }

    void Thread::setCurrentThreadPriority(Priority priority) {
#   if defined(_WIN32)
        SetThreadPriority(GetCurrentThread(), toWindowsPriority(priority));
#   elif defined(__linux__)
        // Linux implements nice values per thread for NPTL. RT64's shader
        // workers request Idle priority specifically so they cannot starve the
        // renderer, but treating that request as a no-op causes severe pipeline
        // compilation stalls and memory pressure on Steam Deck. Lowering a
        // thread's own priority is permitted without elevated privileges.
        if (priority == Thread::Priority::Idle) {
            const id_t threadId = static_cast<id_t>(syscall(SYS_gettid));
            (void)setpriority(PRIO_PROCESS, threadId, 19);
        }
#   elif defined(__APPLE__)
        // Keep the platform default on macOS.
        (void)priority;
#   else
        static_assert(false, "Unimplemented");
#   endif
    }

    void Thread::sleepMilliseconds(uint32_t millis) {
#   if defined(_WIN32)
        // The implementations of std::chrono::sleep_until and sleep_for were affected by changing the system clock backwards in older versions
        // of Microsoft's STL. This was fixed as of Visual Studio 2022 17.9, but to be safe RT64 uses Win32 Sleep directly.
        Sleep(millis);
#   else
        std::this_thread::sleep_for(std::chrono::milliseconds(millis));
#   endif
    }
};
