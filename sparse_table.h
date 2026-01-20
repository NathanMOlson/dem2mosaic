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

#ifndef __SPARSE_TABLE_H__
#define __SPARSE_TABLE_H__

#include <vector>

/**
  * Class representing a sparse table optimized for column wise access.
  */
template <typename C, typename R, typename T>
class SparseTable {
    public:
        typedef std::vector<std::pair<R, T> > Column;

    private:
        std::vector<Column> column_wise_data;

    public:
        SparseTable();
        SparseTable(C cols);

        C cols() const;

        Column const & col(C id) const;

        void set_value(C col, R row, T value);
};

template <typename C, typename R, typename T> C
SparseTable<C, R, T>::cols() const {
    return column_wise_data.size();
}

template <typename C, typename R, typename T> typename SparseTable<C, R, T>::Column const &
SparseTable<C, R, T>::col(C id) const {
    return column_wise_data[id];
}

template <typename C, typename R, typename T>
SparseTable<C, R, T>::SparseTable() {
}

template <typename C, typename R, typename T>
SparseTable<C, R, T>::SparseTable(C cols) {
    column_wise_data.resize(cols);
}

template <typename C, typename R, typename T> void
SparseTable<C, R, T>::set_value(C col, R row, T value) {
    column_wise_data[col].push_back(std::pair<R, T>(row, value));
}

#endif 
