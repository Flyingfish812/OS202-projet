#ifndef MODEL_HPP
#define MODEL_HPP

#include <array>
#include <vector>
#include <map>
#include <cstddef>
#include <cstdint>
#include <mpi.h>

class Model {
public:
    struct LexicoIndices {
        unsigned row;
        unsigned column;
    };

    Model(double t_length, unsigned t_discretization, std::array<double,2> t_wind,
          LexicoIndices t_start_fire_position, double t_max_wind,
          MPI_Comm t_comm, int t_global_row_offset, int t_local_rows);
    ~Model();

    bool update();
    const std::vector<uint8_t>& vegetal_map() const { return m_vegetation_map; }
    const std::vector<uint8_t>& fire_map() const { return m_fire_map; }
    unsigned geometry() const { return m_geometry; }
    std::size_t time_step() const { return m_time_step; }
    std::size_t get_index_from_lexicographic_indices(LexicoIndices t_lexico_indices) const;
    LexicoIndices get_lexicographic_from_index(std::size_t t_global_index) const;
    std::size_t local_index(int local_row, int col) const;

private:
    double m_length;
    unsigned m_geometry;
    double m_distance;
    std::array<double,2> m_wind;
    double m_wind_speed;
    double m_max_wind;

    std::vector<uint8_t> m_vegetation_map;
    std::vector<uint8_t> m_fire_map;
    
    std::map<std::size_t, uint8_t> m_fire_front;
    
    double p1;
    double p2;
    double alphaEastWest, alphaWestEast;
    double alphaSouthNorth, alphaNorthSouth;
    
    std::size_t m_time_step = 0;

    bool distributed = false;
    int local_rows = 0;
    int global_row_offset = 0;
    MPI_Comm comp_comm = MPI_COMM_NULL;
    int comp_rank = 0;
    int comp_size = 0;
};

#endif // MODEL_HPP
