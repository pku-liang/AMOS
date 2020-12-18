#ifndef _PROFILE_H_
#define _PROFILE_H_

#define LINUX
#define OPT_CUDA_PROFILE
#define OPT_TIMER

#ifdef OPT_CUDA_PROFILE
#include <cuda_runtime.h>
#endif

#ifdef LINUX
#include <sys/time.h>
#else
#include <windows.h>
#endif

#define     MAX_OPT_PROFILE_INFO_NUM        1024            //最大profile数量

#ifdef  OPT_TIMER           //打开profile功能
    
//profile信息结构体
struct OPT_PROFILE_INFO
{
#ifdef LINUX
    struct timeval start, end;
#else
    LARGE_INTEGER start, end, freq;
#endif

#ifdef OPT_CUDA_PROFILE
    cudaEvent_t     e_start;
    cudaEvent_t     e_stop;
#endif

    double          time_used;                              //耗时
    int             profile_count;                          //迭代次数
    double          time_used_total;                        //总耗时
    double          calculation;                            //计算量
    double          gflops;                                 //计算量 / 耗时

    double          time_used_max;
    double          time_used_min;

};

//声明全局变量
extern      OPT_PROFILE_INFO                opt_profile_info[];



#define         OPT_PROFILE_TIME_START(num)                         opt_profile_time_start(num)                        
#define         OPT_PROFILE_TIME_STOP(num, str, iter, print_flag)           opt_profile_time_stop(num, str, iter, print_flag)
#define         OPT_PROFILE_TIME_RESET(num)                         opt_profile_time_reset(num)
#define         OPT_PRINT_INFO(str, num)                            opt_print_info(str, num)  
    
/***************************************************************************************************
* 功  能: profile耗时信息
* 参  数:
*               num         指定使用的计时器
* 返回值: 无
***************************************************************************************************/
void opt_profile_time_start(int num);


/***************************************************************************************************
* 功  能: profile耗时信息
* 参  数:
*               num                 指定使用的计时器
*               str                 字符串信息
*               iter                总共迭代次数
*               print_flag          打印信息标志
* 返回值: 无
***************************************************************************************************/
void opt_profile_time_stop(int num, const char *str, int iter, int print_flag);

void opt_print_info(const char *str, int num);

void opt_profile_time_reset(int num);

#else           //关闭profile功能


#define         OPT_PROFILE_TIME_START(num)                                              //profile耗时信息宏
#define         OPT_PROFILE_TIME_STOP(num, str, iter, print_flag)                        //profile耗时信息宏
#define         OPT_PROFILE_TIME_RESET(num)
#define         OPT_PRINT_INFO(num)

#endif



#endif
