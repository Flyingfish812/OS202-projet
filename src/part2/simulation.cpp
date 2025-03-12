#include <string>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <mpi.h>

#include "model.hpp"
#include "display.hpp"

using namespace std::string_literals;
using namespace std::chrono_literals;

std::ofstream logFile;

void initLog() {
    // Get current time
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    // Format time as YYYY-MM-DD_HH-MM-SS
    std::ostringstream oss;
    oss << "logs/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << "_log.txt";
    std::string logFilePath = oss.str();

    logFile.open(logFilePath, std::ios::out | std::ios::trunc);
    if (!logFile) {
        std::cerr << "Failed to open log file: " << logFilePath << std::endl;
        exit(1);
    }
    logFile << "Log initialized at: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;

    // Personalized information
    logFile << "Test for MPI with 2 threads" << std::endl;
}

struct ParamsType
{
    double length{1.};
    unsigned discretization{20u};
    std::array<double,2> wind{0.,0.};
    Model::LexicoIndices start{10u,10u};
};

void analyze_arg( int nargs, char* args[], ParamsType& params )
{
    if (nargs ==0) return;
    std::string key(args[0]);
    if (key == "-l"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une valeur pour la longueur du terrain !" << std::endl;
            exit(EXIT_FAILURE);
        }
        params.length = std::stoul(args[1]);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    auto pos = key.find("--longueur=");
    if (pos < key.size())
    {
        auto subkey = std::string(key,pos+11);
        params.length = std::stoul(subkey);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }

    if (key == "-n"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une valeur pour le nombre de cases par direction pour la discrétisation du terrain !" << std::endl;
            exit(EXIT_FAILURE);
        }
        params.discretization = std::stoul(args[1]);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    pos = key.find("--number_of_cases=");
    if (pos < key.size())
    {
        auto subkey = std::string(key, pos+18);
        params.discretization = std::stoul(subkey);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }

    if (key == "-w"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une paire de valeurs pour la direction du vent !" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string values =std::string(args[1]);
        params.wind[0] = std::stod(values);
        auto pos = values.find(",");
        if (pos == values.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos+1);
        params.wind[1] = std::stod(second_value);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    pos = key.find("--wind=");
    if (pos < key.size())
    {
        auto subkey = std::string(key, pos+7);
        params.wind[0] = std::stoul(subkey);
        auto pos = subkey.find(",");
        if (pos == subkey.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos+1);
        params.wind[1] = std::stod(second_value);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }

    if (key == "-s"s)
    {
        if (nargs < 2)
        {
            std::cerr << "Manque une paire de valeurs pour la position du foyer initial !" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string values =std::string(args[1]);
        params.start.column = std::stod(values);
        auto pos = values.find(",");
        if (pos == values.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la position du foyer initial" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos+1);
        params.start.row = std::stod(second_value);
        analyze_arg(nargs-2, &args[2], params);
        return;
    }
    pos = key.find("--start=");
    if (pos < key.size())
    {
        auto subkey = std::string(key, pos+8);
        params.start.column = std::stoul(subkey);
        auto pos = subkey.find(",");
        if (pos == subkey.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la vitesse" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos+1);
        params.start.row = std::stod(second_value);
        analyze_arg(nargs-1, &args[1], params);
        return;
    }
}

ParamsType parse_arguments( int nargs, char* args[] )
{
    if (nargs == 0) return {};
    if ( (std::string(args[0]) == "--help"s) || (std::string(args[0]) == "-h") )
    {
        std::cout << 
R"RAW(Usage : simulation [option(s)]
  Lance la simulation d'incendie en prenant en compte les [option(s)].
  Les options sont :
    -l, --longueur=LONGUEUR     Définit la taille LONGUEUR (réel en km) du carré représentant la carte de la végétation.
    -n, --number_of_cases=N     Nombre n de cases par direction pour la discrétisation
    -w, --wind=VX,VY            Définit le vecteur vitesse du vent (pas de vent par défaut).
    -s, --start=COL,ROW         Définit les indices I,J de la case où commence l'incendie (milieu de la carte par défaut)
)RAW";
        exit(EXIT_SUCCESS);
    }
    ParamsType params;
    analyze_arg(nargs, args, params);
    return params;
}

bool check_params(ParamsType& params)
{
    bool flag = true;
    if (params.length <= 0)
    {
        std::cerr << "[ERREUR FATALE] La longueur du terrain doit être positive et non nulle !" << std::endl;
        flag = false;
    }

    if (params.discretization <= 0)
    {
        std::cerr << "[ERREUR FATALE] Le nombre de cellules par direction doit être positive et non nulle !" << std::endl;
        flag = false;
    }

    if ( (params.start.row >= params.discretization) || (params.start.column >= params.discretization) )
    {
        std::cerr << "[ERREUR FATALE] Mauvais indices pour la position initiale du foyer" << std::endl;
        flag = false;
    }
    
    return flag;
}

void display_params(ParamsType const& params)
{
    std::cout << "Parametres définis pour la simulation : \n"
              << "\tTaille du terrain : " << params.length << std::endl 
              << "\tNombre de cellules par direction : " << params.discretization << std::endl 
              << "\tVecteur vitesse : [" << params.wind[0] << ", " << params.wind[1] << "]" << std::endl
              << "\tPosition initiale du foyer (col, ligne) : " << params.start.column << ", " << params.start.row << std::endl;
}

int main(int nargs, char* args[])
{
    MPI_Init(&nargs, &args); // Initialisation de MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    initLog();

    // Analyse des paramètres
    ParamsType params;
    params = parse_arguments(nargs-1, &args[1]);
    if (!check_params(params)) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    Model simu(params.length, params.discretization, params.wind, params.start);
    const int size = params.discretization * params.discretization;

    if (rank == 1) { 
        auto global_start_time = std::chrono::high_resolution_clock::now();
        // Initialisation du modèle dans le processus de calcul
        while (simu.update()){
            if ((simu.time_step() & 31) == 0) 
                std::cout << "Time step " << simu.time_step() << "\n===============" << std::endl;
            // Envoi des données sur la végétation et le feu
            MPI_Send(simu.vegetal_map().data(), size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(simu.fire_map().data(), size, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
        }
        
        auto global_end_time = std::chrono::high_resolution_clock::now();
        double global_compute_time = std::chrono::duration<double>(global_end_time - global_start_time).count();
        double average_time_per_iteration = global_compute_time / simu.time_step();
        logFile << "Temps total de simulation : " << global_compute_time << " secondes" << std::endl;
        logFile << "Temps pour étape : " << average_time_per_iteration << " secondes" << std::endl;

        // Envoi du signal de terminaison
        int stop_flag = 1;
        MPI_Send(&stop_flag, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        std::cout << "Simulation terminée." << std::endl;

    } else if (rank == 0) { 
        // Initialisation du processus de rendu
        auto displayer = Displayer::init_instance(params.discretization, params.discretization);
        SDL_Event event;
        bool running = true;
        std::vector<uint8_t> vegetation_data(size), fire_data(size);
        int flag;
        MPI_Status status;
        
        while (running) {
            // Surveillance des messages envoyés par le processus 1 (données ou signal d'arrêt)
            MPI_Iprobe(1, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                // Traitement des données reçues en fonction du tag du message
                if (status.MPI_TAG == 3) { 
                    // Réception du signal d'arrêt
                    MPI_Recv(&flag, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    std::cout << "Signal d'arrêt reçu. Fermeture." << std::endl;
                    running = false;
                    break;
                } else if (status.MPI_TAG == 1) { 
                    // Réception des données sur la végétation et le feu (les tags 1 et 2 doivent être reçus ensemble)
                    MPI_Recv(vegetation_data.data(), size, MPI_BYTE, 1, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(fire_data.data(), size, MPI_BYTE, 1, 2, MPI_COMM_WORLD, &status);

                    // Mesure du temps de mise à jour de l'affichage
                    displayer->update(vegetation_data, fire_data);
                }
            }
    
            // Gestion des événements SDL (fermeture de la fenêtre)
            if (SDL_PollEvent(&event) && event.type == SDL_QUIT) {
                running = false;
            }
            std::this_thread::sleep_for(0.02s); // Évite l'attente active
        }
    }

    // S'assurer que tous les processus terminent correctement
    MPI_Barrier(MPI_COMM_WORLD); // Synchronisation de tous les processus
    MPI_Finalize(); // Fermeture de l'environnement MPI
    logFile.close();
    return EXIT_SUCCESS;
}

