#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <chrono>
#include <vector>
#include "model.hpp"

extern std::ofstream logFile;

namespace
{
    double pseudo_random( std::size_t index, std::size_t time_step )
    {
        std::uint_fast32_t xi = std::uint_fast32_t(index*(time_step+1));
        std::uint_fast32_t r  = (48271*xi)%2147483647;
        return r/2147483646.;
    }

    double log_factor( std::uint8_t value )
    {
        return std::log(1.+value)/std::log(256);
    }
}

Model::Model( double t_length, unsigned t_discretization, std::array<double,2> t_wind,
              LexicoIndices t_start_fire_position, double t_max_wind,
              MPI_Comm t_comm, int t_global_row_offset, int t_local_rows )
    : m_length(t_length),
      m_geometry(t_discretization),
      m_wind(t_wind),
      m_wind_speed(std::sqrt(t_wind[0]*t_wind[0] + t_wind[1]*t_wind[1])),
      m_max_wind(t_max_wind),
      // Région locale avec ghost rows
      m_vegetation_map((t_local_rows+2)*t_discretization, 255u),
      m_fire_map((t_local_rows+2)*t_discretization, 0u),
      m_time_step(0),
      distributed(true),
      local_rows(t_local_rows),
      global_row_offset(t_global_row_offset)
{
    MPI_Comm_dup(t_comm, &comp_comm);
    MPI_Comm_rank(comp_comm, &comp_rank);
    MPI_Comm_size(comp_comm, &comp_size);

    m_distance = t_length / double(t_discretization);

    int start_global_row = t_start_fire_position.row;
    if ( start_global_row >= t_global_row_offset && start_global_row < t_global_row_offset + t_local_rows )
    {
        int local_row = start_global_row - t_global_row_offset + 1;
        int index = local_row * m_geometry + t_start_fire_position.column;
        m_fire_map[index] = 255u;
        m_fire_front[index] = 255u;
    }

    constexpr double alpha0 = 4.52790762e-01;
    constexpr double alpha1 = 9.58264437e-04;
    constexpr double alpha2 = 3.61499382e-05;

    if (m_wind_speed < t_max_wind)
        p1 = alpha0 + alpha1*m_wind_speed + alpha2*(m_wind_speed*m_wind_speed);
    else 
        p1 = alpha0 + alpha1*t_max_wind + alpha2*(t_max_wind*t_max_wind);
    p2 = 0.3;

    if (m_wind[0] > 0)
    {
        alphaEastWest = std::abs(m_wind[0]/t_max_wind)+1;
        alphaWestEast = 1.-std::abs(m_wind[0]/t_max_wind);    
    }
    else
    {
        alphaWestEast = std::abs(m_wind[0]/t_max_wind)+1;
        alphaEastWest = 1. - std::abs(m_wind[0]/t_max_wind);
    }

    if (m_wind[1] > 0)
    {
        alphaSouthNorth = std::abs(m_wind[1]/t_max_wind) + 1;
        alphaNorthSouth = 1. -std::abs(m_wind[1]/t_max_wind);
    }
    else
    {
        alphaNorthSouth = std::abs(m_wind[1]/t_max_wind) + 1;
        alphaSouthNorth = 1. -std::abs(m_wind[1]/t_max_wind);
    }
}

std::size_t Model::local_index( int local_row, int col ) const
{
    return local_row * m_geometry + col;
}

// 对非分布式版本保持原有映射函数不变
std::size_t   
Model::get_index_from_lexicographic_indices( LexicoIndices t_lexico_indices  ) const
{
    return t_lexico_indices.row*this->geometry() + t_lexico_indices.column;
}

auto 
Model::get_lexicographic_from_index( std::size_t t_global_index ) const -> LexicoIndices
{
    LexicoIndices ind_coords;
    ind_coords.row    = t_global_index/this->geometry();
    ind_coords.column = t_global_index%this->geometry();
    return ind_coords;
}

bool Model::update()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Echanger le ghost row
    // Haut : Si existe un voisin au-dessus, envoyer la première ligne active (ligne 1) à ce voisin 
    // et recevoir la dernière ligne de ce voisin pour la placer dans le ghost row supérieur (ligne 0)
    if(comp_rank > 0) {
        MPI_Sendrecv(&m_fire_map[local_index(1,0)], m_geometry, MPI_BYTE, comp_rank-1, 0,
                        &m_fire_map[local_index(0,0)], m_geometry, MPI_BYTE, comp_rank-1, 1,
                        comp_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_vegetation_map[local_index(1,0)], m_geometry, MPI_BYTE, comp_rank-1, 2,
                        &m_vegetation_map[local_index(0,0)], m_geometry, MPI_BYTE, comp_rank-1, 3,
                        comp_comm, MPI_STATUS_IGNORE);
    }
    // Bas : Si existe un voisin en dessous, envoyer la dernière ligne active (ligne local_rows) à ce voisin
    // et recevoir la première ligne de ce voisin pour la placer dans le ghost row inférieur (ligne local_rows+1)
    if(comp_rank < comp_size-1) {
        MPI_Sendrecv(&m_fire_map[local_index(local_rows,0)], m_geometry, MPI_BYTE, comp_rank+1, 1,
                        &m_fire_map[local_index(local_rows+1,0)], m_geometry, MPI_BYTE, comp_rank+1, 0,
                        comp_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_vegetation_map[local_index(local_rows,0)], m_geometry, MPI_BYTE, comp_rank+1, 3,
                        &m_vegetation_map[local_index(local_rows+1,0)], m_geometry, MPI_BYTE, comp_rank+1, 2,
                        comp_comm, MPI_STATUS_IGNORE);
    }

    // Enregistrement local des mises à jour de chaque thread
    std::vector<uint8_t> new_fire_map = m_fire_map;
    // Une boucle sur les cases du tableau
    for(int i = 0; i <= local_rows+1; ++i) {
        for(int j = 0; j < (int)m_geometry; ++j) {
            std::size_t idx = local_index(i,j);
            uint8_t fire_value = m_fire_map[idx];
            if(fire_value > 0) {
                double power = log_factor(fire_value);
                
                if(i-1 >= 0) {
                    std::size_t idx_up = local_index(i-1,j);
                    double tirage = pseudo_random(idx + m_time_step, m_time_step);
                    double green_power = m_vegetation_map[idx_up];
                    double correction = power * log_factor(green_power);
                    if(tirage < alphaNorthSouth * p1 * correction) {
                        new_fire_map[idx_up] = 255;
                    }
                }
                
                if(i+1 <= local_rows+1) {
                    std::size_t idx_down = local_index(i+1,j);
                    double tirage = pseudo_random(idx * 13427 + m_time_step, m_time_step);
                    double green_power = m_vegetation_map[idx_down];
                    double correction = power * log_factor(green_power);
                    if(tirage < alphaSouthNorth * p1 * correction) {
                        new_fire_map[idx_down] = 255;
                    }
                }
                
                if(j-1 >= 0) {
                    std::size_t idx_left = local_index(i,j-1);
                    double tirage = pseudo_random(idx * 13427 * 13427 + m_time_step, m_time_step);
                    double green_power = m_vegetation_map[idx_left];
                    double correction = power * log_factor(green_power);
                    if(tirage < alphaWestEast * p1 * correction) {
                        new_fire_map[idx_left] = 255;
                    }
                }
                
                if(j+1 < (int)m_geometry) {
                    std::size_t idx_right = local_index(i,j+1);
                    double tirage = pseudo_random(idx * 13427 * 13427 * 13427 + m_time_step, m_time_step);
                    double green_power = m_vegetation_map[idx_right];
                    double correction = power * log_factor(green_power);
                    if(tirage < alphaEastWest * p1 * correction) {
                        new_fire_map[idx_right] = 255;
                    }
                }

                if(fire_value == 255) {
                    double tirage = pseudo_random(idx * 52513 + m_time_step, m_time_step);
                    if(tirage < p2) {
                        new_fire_map[idx] = fire_value >> 1;
                    }
                } else {
                    new_fire_map[idx] = fire_value >> 1;
                }
                
                if(m_vegetation_map[idx] > 0)
                    m_vegetation_map[idx] -= 1;
            }
        }
    }
    m_fire_map = new_fire_map;
    m_time_step += 1;

    bool active = false;
    for(int i = 1; i <= local_rows; ++i) {
        for(int j = 0; j < (int)m_geometry; ++j) {
            if(m_fire_map[local_index(i,j)] > 0) {
                active = true;
                break;
            }
        }
        if(active) break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    logFile << "Temps pour une étape (distribué): " << elapsed.count() << " secondes" << std::endl;
    return active;
}

Model::~Model() {
    if(distributed) {
        MPI_Comm_free(&comp_comm);
    }
}
