#include <vector>
#include <tuple>


// Vector saves w, h, c, n,
//              k, filter_w(s), filter_h(r), pad_w,
//              pad_h, wstride, hstride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int>>
    yolo_v1 = {
        std::make_tuple(448, 448, 3, 1, 64, 7, 7, 3, 3, 2, 2),
        std::make_tuple(112, 112, 64, 1, 192, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 192, 1, 128, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 256, 1, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 256, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 512, 1, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 512, 1, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 1, 1024, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 1024, 1, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 512, 1, 1024, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 1024, 1, 1024, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 1024, 1, 1024, 3, 3, 1, 1, 2, 2),
        std::make_tuple(7, 7, 1024, 1, 1024, 3, 3, 1, 1, 1, 1)
};

// Vector saves w, h, c, n,
//              k, filter_w(s), filter_h(r), pad_w,
//              pad_h, wstride, hstride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int>>
    res18_v1 = {
        std::make_tuple(224, 224, 3, 1, 64, 7, 7, 3, 3, 2, 2),
        std::make_tuple(56, 56, 64, 1, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 64, 1, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 64, 1, 128, 3, 3, 1, 1, 2, 2),
        std::make_tuple(56, 56, 64, 1, 128, 1, 1, 0, 0, 2, 2),
        std::make_tuple(28, 28, 128, 1, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 128, 1, 256, 3, 3, 1, 1, 2, 2),
        std::make_tuple(28, 28, 128, 1, 256, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 256, 1, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 256, 1, 512, 3, 3, 1, 1, 2, 2),
        std::make_tuple(14, 14, 256, 1, 512, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
};

// Vector saves w, h, c, n,
//              k, filter_w(s), filter_h(r), pad_w,
//              pad_h, wstride, hstride
//              dilation_h, dilation_w
// ResNet-50 strided conv -> dilated conv
// which is used in DeepLab detection model
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int, unsigned int> >
    conv_dilated = {
                    //   w,   h,   c, n,  k,  s, r,pw,ph,sw,sh,dw,dh
        std::make_tuple(56, 56, 256, 1, 512, 1, 1, 0, 0, 1, 1, 2, 2),
        std::make_tuple(56, 56, 256, 1, 128, 1, 1, 0, 0, 1, 1, 2, 2),
        std::make_tuple(28, 28, 512, 1, 1024, 1, 1, 0, 0, 1, 1, 2, 2),
        std::make_tuple(28, 28, 512, 1, 256, 1, 1, 0, 0, 1, 1, 2, 2),
        std::make_tuple(14, 14, 1024, 1, 2048, 1, 1, 0, 0, 1, 1, 2, 2),
        std::make_tuple(14, 14, 1024, 1, 512, 1, 1, 0, 0, 1, 1, 2, 2),
};


// Vector saves l, c, n, k, filter, pad, stride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int>>
    conv_1d = {
        //              l,    c, n,  k, filter, pad, stride
        std::make_tuple(448,  3, 1, 64,     7,    0,      1),
        std::make_tuple(112, 64, 1, 192,    3,    0,      1)
};



std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int>>
    conv_3d = {
                    //  d, kern_d,  w,  h, c,n=1,  k,  s,   r, pad_d, pad_w, pad_h, dstride, wstride, hstride
        std::make_tuple(4,    3,  64, 64,  4,  1, 32,  16, 16,     0,     0,    0,       1,        1,      1),
        std::make_tuple(4,    3,  32, 32,  4,  1, 16,  16, 16,     0,     0,    0,       1,        1,      1)
};


// Vector saves w, h, c, n,
//              k, filter_w(s), filter_h(r), pad_w,
//              pad_h, wstride, hstride, groupcnt
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>>
    alex_grouped = { 
		  // modify from yolo_v1 to satisfy: groupcnt == c, and k is multiple of c
		  //                w,   h,  c,  n,     k, fw,  fh, pw, ph, ws, hs, groupcnt
		  std::make_tuple(27, 27, 96,  1,  256,  5,   5,  2,  2,  1,  1,  2),
		  std::make_tuple(13, 13, 384,  1,  384,  3,   3,  1,  1,  1,  1,  2),
		  std::make_tuple(13,  13, 384,  1,  256,  3,   3,  1,  1,  1,  1,  2),
};


// Vector saves w, h, c, n,
//              k, filter_w(s), filter_h(r), pad_w,
//              pad_h, wstride, hstride, groupcnt
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>>
    shuffle_v1 = { 
		  // modify from yolo_v1 to satisfy: groupcnt == c, and k is multiple of c
		  //                w,   h,  c,  n,     k, fw,  fh, pw, ph, ws, hs, groupcnt
		  std::make_tuple(224, 224, 3,  1,  24,  3,   3,  1,  1,  2,  2,  3),
		// cudnn doesn't support this one
        //   std::make_tuple(56, 56, 24,  1,  54,  1,   1,  0,  0,  1,  1,  3),
          std::make_tuple(28, 28, 54,  1,  216,  1,   1,  0,  0,  1,  1,  3),
          std::make_tuple(28, 28, 240,  1,  60,  1,   1,  0,  0,  1,  1,  3),
          std::make_tuple(28, 28, 60,  1,  240,  1,   1,  0,  0,  1,  1,  3),
};


// Vector saves w, h, c, n,
//              k, filter_w(s), filter_h(r), pad_w,
//              pad_h, wstride, hstride, groupcnt
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>>
    depth_wise = { 
		  // modify from yolo_v1 to satisfy: groupcnt == c, and k is multiple of c
		  //                w,   h,  c,  n,     k, fw,  fh, pw, ph, ws, hs, groupcnt
		  std::make_tuple(112, 112, 32,  1,  32*1,  3,   3,  1,  1,  1,  1,  32),
		  std::make_tuple(112, 112, 16,  1,  16*6,  3,   3,  1,  1,  2,  2,  16),
		  std::make_tuple( 56,  56, 24,  1,  24*6,  3,   3,  1,  1,  2,  2,  24),
		  std::make_tuple( 28,  28, 32,  1,  32*6,  3,   3,  1,  1,  2,  2,  32),
		  std::make_tuple( 14,  14, 64,  1,  64*6,  3,   3,  1,  1,  1,  1,  64),
		  std::make_tuple( 14,  14, 96,  1,  96*6,  3,   3,  1,  1,  2,  2,  96),
		  std::make_tuple(  7,   7,160,  1, 160*6,  3,   3,  1,  1,  1,  1, 160)
};


// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> gemm_set = {
    // std::make_tuple(1024, 6000, 2816, false, false),
    // std::make_tuple(1024, 6000, 2048, false, false),
    // std::make_tuple(1024, 6000, 2560, false, false),
    // std::make_tuple(1024, 6000, 1536, false, false),
    // std::make_tuple(512, 4, 512, false, false),
    // std::make_tuple(1024, 4, 512, false, false)
    std::make_tuple(16, 512, 128, false, false),
    std::make_tuple(1024, 16, 256, false, false),
    std::make_tuple(256, 1024, 256, false, false),
    std::make_tuple(512, 256, 16, false, false),
    std::make_tuple(1024, 1024, 1024, false, false)
};

// Vector saves m, n, trans
std::vector<std::tuple<int, int, bool>> gemv_noTC = {
    std::make_tuple(1024, 256, false),
    std::make_tuple(512, 1024, false)
};

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> gemv_to_gemm = {
    std::make_tuple(1024, 1, 2816, false, false),
    std::make_tuple(1024, 1, 2048, false, false),
    std::make_tuple(1024, 1, 2560, false, false),
    std::make_tuple(1024, 1, 1536, false, false),
    std::make_tuple(512, 1, 512, false, false),
    std::make_tuple(1024, 1, 512, false, false)
};