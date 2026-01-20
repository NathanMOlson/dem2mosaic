/** DEM2mosaic: convert a DEM and an OpenSfM reconstruction file to a orthomosaic.
 Copyright (C) 2025-2026 Lab 308, LLC

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **/

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