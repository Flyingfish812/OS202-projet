#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "model.hpp"
#include <omp.h>
#include <chrono>

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
              LexicoIndices t_start_fire_position, double t_max_wind )
    :   m_length(t_length),
        m_distance(-1),
        m_geometry(t_discretization),
        m_wind(t_wind),
        m_wind_speed(std::sqrt(t_wind[0]*t_wind[0] + t_wind[1]*t_wind[1])),
        m_max_wind(t_max_wind),
        m_vegetation_map(t_discretization*t_discretization, 255u),
        m_fire_map(t_discretization*t_discretization, 0u)
{
    if (t_discretization == 0)
    {
        throw std::range_error("Le nombre de cases par direction doit être plus grand que zéro.");
    }
    m_distance = m_length/double(m_geometry);
    auto index = get_index_from_lexicographic_indices(t_start_fire_position);
    m_fire_map[index] = 255u;
    m_fire_front[index] = 255u;

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
        alphaNorthSouth = 1. - std::abs(m_wind[1]/t_max_wind);
    }
    else
    {
        alphaNorthSouth = std::abs(m_wind[1]/t_max_wind) + 1;
        alphaSouthNorth = 1. - std::abs(m_wind[1]/t_max_wind);
    }
}
// --------------------------------------------------------------------------------------------------------------------
bool Model::update() {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Stockage local des threads pour collecter les mises à jour de chaque thread
    int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<std::size_t, std::uint8_t>> thread_local_next_fronts(num_threads);
    std::vector<std::vector<std::size_t>> thread_local_erased_keys(num_threads);
    std::vector<double> thread_times(num_threads, 0.0);

    // Obtenir les indices de tous les points de feu pour assurer un ordre de parcours cohérent
    std::vector<std::size_t> keys;
    keys.reserve(m_fire_front.size());
    for (const auto& f : m_fire_front) {
        keys.push_back(f.first);
    }

    // Traitement parallèle de chaque point de feu
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < keys.size(); i++) {
        int tid = omp_get_thread_num();  // Obtenir l'ID du thread actuel
        auto thread_start_time = std::chrono::high_resolution_clock::now();
        auto& local_next_front = thread_local_next_fronts[tid];  // Conteneur local de mise à jour du thread
        auto& local_erased = thread_local_erased_keys[tid];      // Conteneur local des clés à supprimer

        std::size_t fire_index = keys[i];
        auto it = m_fire_front.find(fire_index);
        if (it == m_fire_front.end()) continue;
        auto f = *it;

        // Obtenir les coordonnées et l'intensité du point de feu actuel
        LexicoIndices coord = get_lexicographic_from_index(f.first);
        double power = log_factor(f.second);

        // Vérifier et mettre à jour les cellules adjacentes
        if (coord.row < m_geometry - 1) {
            double tirage = pseudo_random(f.first + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first + m_geometry];
            double correction = power * log_factor(green_power);
            if (tirage < alphaSouthNorth * p1 * correction) {
                m_fire_map[f.first + m_geometry] = 255;  // Mise à jour du point de feu
                local_next_front[f.first + m_geometry] = 255;  // Enregistrement local du nouveau point de feu
            }
        }

        if (coord.row > 0) {
            double tirage = pseudo_random(f.first * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first - m_geometry];
            double correction = power * log_factor(green_power);
            if (tirage < alphaNorthSouth * p1 * correction) {
                m_fire_map[f.first - m_geometry] = 255;
                local_next_front[f.first - m_geometry] = 255;
            }
        }

        if (coord.column < m_geometry - 1) {
            double tirage = pseudo_random(f.first * 13427 * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first + 1];
            double correction = power * log_factor(green_power);
            if (tirage < alphaEastWest * p1 * correction) {
                m_fire_map[f.first + 1] = 255;
                local_next_front[f.first + 1] = 255;
            }
        }

        if (coord.column > 0) {
            double tirage = pseudo_random(f.first * 13427 * 13427 * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first - 1];
            double correction = power * log_factor(green_power);
            if (tirage < alphaWestEast * p1 * correction) {
                m_fire_map[f.first - 1] = 255;
                local_next_front[f.first - 1] = 255;
            }
        }

        // Gérer la diminution de l'intensité du feu
        if (f.second == 255) {
            double tirage = pseudo_random(f.first * 52513 + m_time_step, m_time_step);
            if (tirage < p2) {
                m_fire_map[f.first] >>= 1;
                local_next_front[f.first] = f.second >> 1;  // Réduction de l'intensité du feu
            } else {
                local_next_front[f.first] = f.second;  // L'intensité du feu reste inchangée
            }
        } else {
            m_fire_map[f.first] >>= 1;
            local_next_front[f.first] = f.second >> 1;  // Continuer à réduire l'intensité du feu
        }

        // Si l'intensité du feu atteint 0, enregistrer la clé pour suppression
        if (local_next_front[f.first] == 0) {
            local_erased.push_back(f.first);
        }

        auto thread_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = thread_end_time - thread_start_time;
        thread_times[tid] = elapsed.count();  // 记录线程时间
    }

    // Fusionner en série toutes les mises à jour des threads
    auto next_front = m_fire_front;
    for (auto& local_front : thread_local_next_fronts) {
        for (auto& [key, val] : local_front) {
            next_front[key] = val;
        }
    }

    // Gérer les suppressions
    for (auto& local_erased : thread_local_erased_keys) {
        for (auto key : local_erased) {
            next_front.erase(key);
            m_fire_map[key] = 0;
        }
    }

    // Mettre à jour l'état des points de feu
    m_fire_front = next_front;

    // Mettre à jour l'état de la végétation
    for (auto f : m_fire_front) {
        if (m_vegetation_map[f.first] > 0) {
            m_vegetation_map[f.first] -= 1;
        }
    }

    // Mettre à jour le pas de temps
    m_time_step += 1;

    // Sauvegarder l'état du feu et de la végétation tous les 100 pas
    if (m_time_step % 100 == 0) {
        std::ostringstream fire_filename, vegetation_filename;
        fire_filename << "outputs/fire_map_step_" << m_time_step << ".txt";
        vegetation_filename << "outputs/vegetation_map_step_" << m_time_step << ".txt";

        std::ofstream fire_file(fire_filename.str());
        std::ofstream vegetation_file(vegetation_filename.str());

        if (fire_file && vegetation_file) {
            for (unsigned row = 0; row < m_geometry; ++row) {
                for (unsigned col = 0; col < m_geometry; ++col) {
                    std::size_t idx = row * m_geometry + col;
                    fire_file << (int)m_fire_map[idx] << " ";
                    vegetation_file << (int)m_vegetation_map[idx] << " ";
                }
                fire_file << "\n";
                vegetation_file << "\n";
            }
            logFile << "Sauvegarde des cartes du feu et de la végétation à l'étape " << m_time_step << std::endl;
        } else {
            logFile << "Erreur : Impossible d'ouvrir les fichiers de sortie pour l'étape " << m_time_step << std::endl;
        }
    }

    // Afficher le temps d'exécution
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double elapsed_time = elapsed.count();

    for (int i = 0; i < num_threads; i++) {
        logFile << "Thread " << i << " time: " << thread_times[i] << "s\n";
    }
    logFile << "Global execution time: " << elapsed_time << "s\n";

    // Retourner si des points de feu sont encore actifs
    return !m_fire_front.empty();
}

// ====================================================================================================================
std::size_t   
Model::get_index_from_lexicographic_indices( LexicoIndices t_lexico_indices  ) const
{
    return t_lexico_indices.row*this->geometry() + t_lexico_indices.column;
}
// --------------------------------------------------------------------------------------------------------------------
auto 
Model::get_lexicographic_from_index( std::size_t t_global_index ) const -> LexicoIndices
{
    LexicoIndices ind_coords;
    ind_coords.row    = t_global_index/this->geometry();
    ind_coords.column = t_global_index%this->geometry();
    return ind_coords;
}
