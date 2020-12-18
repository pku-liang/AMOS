#ifdef OPT_CUDA_PROFILE
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include "profile.h"

#ifdef OPT_TIMER
//profile全局变量
OPT_PROFILE_INFO    opt_profile_info[MAX_OPT_PROFILE_INFO_NUM];

/***************************************************************************************************
* 功  能: profile耗时信息
* 参  数:
*               num         指定使用的计时器
* 返回值: 无
***************************************************************************************************/
void opt_profile_time_start(int num)
{
#ifdef OPT_TIMER

#ifdef OPT_CUDA_PROFILE
    cudaDeviceSynchronize();
#endif

#ifdef LINUX
#else
    QueryPerformanceFrequency(&opt_profile_info[num].freq);
#endif

    //cudaDeviceSynchronize();

#ifdef LINUX
    gettimeofday(&(opt_profile_info[num].start), NULL);
#else
    QueryPerformanceCounter(&opt_profile_info[num].start);
#endif

#endif
}

void opt_profile_time_reset(int num)
{
    //memset(opt_profile_info + num, 0, sizeof(OPT_PROFILE_INFO));

    //cudaEventCreate(&opt_profile_info[num].e_start);
    //cudaEventCreate(&opt_profile_info[num].e_stop);
    //cudaEventDestroy(opt_profile_info[num].e_start);
    //cudaEventDestroy(opt_profile_info[num].e_stop);

    opt_profile_info[num].time_used             = 0.0;
    opt_profile_info[num].profile_count = 0;
    opt_profile_info[num].time_used_total       = 0.0;
    opt_profile_info[num].calculation           = 0.0;
    opt_profile_info[num].gflops                = 0.0;
    opt_profile_info[num].time_used_max         = -100000000.0;
    opt_profile_info[num].time_used_min         = 9999999999.0;
}    



void opt_print_info(const char *str, int num)
{

    printf("%10s: avg: %10f max: %10f min: %10f total: %10f iter: %10d\n",
        str,
        opt_profile_info[num].time_used_total / opt_profile_info[num].profile_count,
        opt_profile_info[num].time_used_max,
        opt_profile_info[num].time_used_min,
        opt_profile_info[num].time_used_total,
        opt_profile_info[num].profile_count);
}

/***************************************************************************************************
* 功  能: profile耗时信息
* 参  数:
*               num                 指定使用的计时器
*               str                 字符串信息
*               iter                总共迭代次数
*               print_flag          打印信息标志
* 返回值: 无
***************************************************************************************************/
void opt_profile_time_stop(int num, const char *str, int iter, int print_flag)
{
#ifdef OPT_TIMER
    double             time_used;

#ifdef OPT_CUDA_PROFILE
    cudaDeviceSynchronize();
#endif

#ifdef LINUX
    gettimeofday(&(opt_profile_info[num].end), NULL);
#else
    QueryPerformanceCounter(&opt_profile_info[num].end);
#endif


#ifdef LINUX
    time_used = 1.0 * (opt_profile_info[num].end.tv_sec - opt_profile_info[num].start.tv_sec) * 1000.0 + 1.0 * (opt_profile_info[num].end.tv_usec - opt_profile_info[num].start.tv_usec) / 1000.0;
#else
    time_used = 1.0 * (opt_profile_info[num].end.QuadPart - opt_profile_info[num].start.QuadPart) / opt_profile_info[num].freq.QuadPart * 1000;
#endif

    time_used /= iter;

    opt_profile_info[num].time_used_max = opt_profile_info[num].time_used_max > time_used ? opt_profile_info[num].time_used_max : time_used;
    opt_profile_info[num].time_used_min = opt_profile_info[num].time_used_min < time_used ? opt_profile_info[num].time_used_min : time_used;

    opt_profile_info[num].time_used        = time_used;
    opt_profile_info[num].time_used_total += time_used;
    opt_profile_info[num].profile_count   += 1;

    if (print_flag)
    {
        printf("%s time_used: %f avg: %f max: %f min: %f total: %f iter: %d\n", 
               str, 
               time_used, 
               opt_profile_info[num].time_used_total / opt_profile_info[num].profile_count, 
               opt_profile_info[num].time_used_max,
               opt_profile_info[num].time_used_min,
               opt_profile_info[num].time_used_total,
               opt_profile_info[num].profile_count);
    }
#else
    opt_profile_info[num].time_used = 0.0;
#endif
}

#endif
