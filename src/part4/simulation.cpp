#include <string>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <vector>
#include <mpi.h>

#include "model.hpp"
#include "display.hpp"

using namespace std::string_literals;
using namespace std::chrono_literals;

std::ofstream logFile;

void initLog(int rank) {
    // 获取当前时间
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    // 格式化时间为 YYYY-MM-DD_HH-MM-SS
    std::ostringstream oss;
    oss << "logs/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << "_thread_" << rank << "_log.txt";
    std::string logFilePath = oss.str();

    logFile.open(logFilePath, std::ios::out | std::ios::trunc);
    if (!logFile) {
        std::cerr << "Failed to open log file: " << logFilePath << std::endl;
        exit(1);
    }
    logFile << "Log initialized at: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;

    logFile << "Part 4 : Test for MPI with multiple compute processes" << std::endl;
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
        std::string values = std::string(args[1]);
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
        params.wind[0] = std::stod(subkey);
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
        std::string values = std::string(args[1]);
        params.start.column = std::stoul(values);
        auto pos = values.find(",");
        if (pos == values.size())
        {
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la position du foyer initial" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(values, pos+1);
        params.start.row = std::stoul(second_value);
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
            std::cerr << "Doit fournir deux valeurs séparées par une virgule pour définir la position du foyer initial" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto second_value = std::string(subkey, pos+1);
        params.start.row = std::stoul(second_value);
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
    MPI_Init(&nargs, &args);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    initLog(world_rank);
    ParamsType params = parse_arguments(nargs-1, &args[1]);
    if (!check_params(params)) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // 将 MPI_COMM_WORLD 分裂成两个子通信组：rank 0 为渲染，其余为计算
    int color = (world_rank == 0) ? 0 : 1;
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &new_comm);

    // 在 main() 函数中
    if (world_rank == 0) {
        // 渲染进程部分
        int global_rows = params.discretization;
        int global_cols = params.discretization;
        std::vector<uint8_t> global_vegetation(global_rows * global_cols, 255);
        std::vector<uint8_t> global_fire(global_rows * global_cols, 0);

        // 计算计算进程数量（world_size - 1）以及各条带行数和全局偏移
        int compute_procs = world_size - 1;
        std::vector<int> local_rows(compute_procs, global_rows / compute_procs);
        int remainder = global_rows % compute_procs;
        for (int i = 0; i < remainder; ++i) {
            local_rows[i] += 1;
        }
        std::vector<int> offsets(compute_procs, 0);
        for (int i = 1; i < compute_procs; ++i) {
            offsets[i] = offsets[i-1] + local_rows[i-1];
        }

        auto displayer = Displayer::init_instance(global_cols, global_rows);
        SDL_Event event;
        bool running = true;
        MPI_Status status;
        while (running) {
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                if (status.MPI_TAG == 3) {
                    // 接收到计算进程发来的终止信号
                    int term;
                    MPI_Recv(&term, 1, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // 向所有计算进程发送结束命令（tag 4）
                    for (int dest = 1; dest < world_size; ++dest) {
                        int dummy = 1;
                        MPI_Send(&dummy, 1, MPI_INT, dest, 4, MPI_COMM_WORLD);
                    }
                    std::cout << "Termination signal received. Exiting rendering loop." << std::endl;
                    running = false;
                    continue;
                } else if (status.MPI_TAG == 1) {
                    // 接收来自计算进程的局部数据
                    int src = status.MPI_SOURCE;
                    int proc_index = src - 1;
                    int rows = local_rows[proc_index];
                    int count = rows * global_cols;
                    std::vector<uint8_t> local_vegetation(count);
                    std::vector<uint8_t> local_fire(count);
                    MPI_Recv(local_vegetation.data(), count, MPI_BYTE, src, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(local_fire.data(), count, MPI_BYTE, src, 2, MPI_COMM_WORLD, &status);
                    int offset = offsets[proc_index] * global_cols;
                    std::copy(local_vegetation.begin(), local_vegetation.end(), global_vegetation.begin() + offset);
                    std::copy(local_fire.begin(), local_fire.end(), global_fire.begin() + offset);
                }
            }
            displayer->update(global_vegetation, global_fire);

            if (SDL_PollEvent(&event) && event.type == SDL_QUIT) {
                running = false;
                // 用户主动退出时也向所有计算进程发送终止命令
                for (int dest = 1; dest < world_size; ++dest) {
                    int dummy = 1;
                    MPI_Send(&dummy, 1, MPI_INT, dest, 4, MPI_COMM_WORLD);
                }
            }
            // std::this_thread::sleep_for(20ms);
        }
    } else {
        // 计算进程部分
        int comp_rank, comp_size;
        MPI_Comm_rank(new_comm, &comp_rank);
        MPI_Comm_size(new_comm, &comp_size);
        int global_rows = params.discretization;
        int global_cols = params.discretization;
        int base_rows = global_rows / comp_size;
        int remainder = global_rows % comp_size;
        int local_domain_rows = (comp_rank < remainder) ? base_rows + 1 : base_rows;
        int global_offset = (comp_rank < remainder) ? comp_rank * (base_rows + 1)
                                                    : remainder * (base_rows + 1) + (comp_rank - remainder)*base_rows;
        Model simu(params.length, params.discretization, params.wind, params.start, 10.0, new_comm, global_offset, local_domain_rows);
        int size = local_domain_rows * global_cols; // active 区域大小（不含 ghost 行）

        bool local_active = true;
        while (true) {
            // 更新本地区域火势传播
            local_active = simu.update();
            // 全局判断是否还有火势存在（通过 new_comm 内的所有计算进程）
            bool global_active = false;
            MPI_Allreduce(&local_active, &global_active, 1, MPI_C_BOOL, MPI_LOR, new_comm);
            if (!global_active) {
                break;
            }
            // 发送当前 active 区域数据（跳过 ghost 行）到渲染进程
            MPI_Send(simu.vegetal_map().data() + global_cols, size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(simu.fire_map().data() + global_cols, size, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
        }
        
        // 所有计算进程在此处等待同步
        MPI_Barrier(new_comm);
        // 由 new_comm 内指定的进程发送终止信号到渲染进程
        if (comp_rank == 0) {
            int stop_flag = 1;
            MPI_Send(&stop_flag, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }
        // 等待来自渲染进程的终止命令（tag 4）
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Compute process " << world_rank << " received termination command, exiting." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    logFile.close();
    return EXIT_SUCCESS;
}
