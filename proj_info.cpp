#include "proj_info.h"
#include <fstream>

pcstype_t zone_number_to_pcstype(int zone)
{
    if (zone >= 0)
    {
        return (pcstype_t)(32600 + zone);
    }
    return (pcstype_t)(32700 - zone);
}

ProjInfo parse_georef_file(const std::filesystem::path &filepath)
{
    ProjInfo proj;
    std::ifstream f(filepath);
    std::string str;
    std::getline(f, str);

    const std::string expected_start("WGS84 UTM ");

    if (str.substr(0, expected_start.size()) != expected_start)
    {
        throw std::invalid_argument("georeference file \"" + filepath.string() + "\" started with " + str + ", expected " + expected_start);
    }

    std::string utm_zone = str.substr(expected_start.size());

    char hemisphere = utm_zone.back();
    utm_zone.pop_back();
    int zone = stoi(utm_zone);
    if (hemisphere == 'S')
    {
        zone *= -1;
    }
    else if (hemisphere != 'N')
    {
        throw std::invalid_argument("georeference file \"" + filepath.string() + "\" had invalid UTM zone hemisphere \"" + hemisphere + "\"");
    }

    f >> proj.x_offset >> proj.y_offset;

    proj.crs = zone_number_to_pcstype(zone);

    return proj;
}