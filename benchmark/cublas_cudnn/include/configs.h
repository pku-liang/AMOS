#include <vector>
#include <tuple>


// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride,
// hstride
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