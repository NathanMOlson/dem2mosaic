#include "save_geotiff.h"
#include "geotiff/xtiffio.h"
#include "geotiff/geotiffio.h"

int save_geotiff(const std::filesystem::path &filepath, const cv::Mat &img)
{
    if (img.depth() != CV_8U && img.depth() != CV_16U)
    {
        return -1;
    }

    TIFF *tif = nullptr;
    GTIF *gtif = nullptr;

    tif = XTIFFOpen(filepath.c_str(), "w");
    if (!tif)
    {
        return -2;
    }

    gtif = GTIFNew(tif);
    if (!gtif)
    {
        XTIFFClose(tif);
        return -3;
    }

    constexpr uint32_t tile_width = 256;

    size_t bits_per_sample = (img.depth() == CV_16U || img.depth() == CV_16S) ? 16 : 8;

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.cols);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.rows);
    TIFFSetField(tif, TIFFTAG_TILEWIDTH, tile_width);
    TIFFSetField(tif, TIFFTAG_TILELENGTH, tile_width);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, img.channels());
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    if (img.depth() == CV_16U || img.depth() == CV_8U)
    {
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    }
    else if (img.depth() == CV_16S)
    {
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
    }
    if (img.channels() == 3)
    {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    }
    else
    {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    }
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);

    size_t bits_per_pixel = bits_per_sample * img.channels();
    size_t bytes_per_line = bits_per_pixel / 8 * img.cols;

    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, bytes_per_line));

    cv::Mat chip(tile_width, tile_width, img.type());
    for (int i = 0; i < img.rows; i += tile_width)
    {
        for (int j = 0; j < img.cols; j += tile_width)
        {
            int w = std::min((int)tile_width, img.cols - j);
            int h = std::min((int)tile_width, img.rows - i);
            if (w < (int)tile_width || h < (int)tile_width)
            {
                chip = cv::Mat::zeros(tile_width, tile_width, img.type());
            }
            img(cv::Rect(j, i, w, h)).copyTo(chip(cv::Rect(0, 0, w, h)));
            TIFFWriteTile(tif, chip.data, j, i, 0, 0);
        }
    }
    GTIFFree(gtif);
    XTIFFClose(tif);

    return 0;
}