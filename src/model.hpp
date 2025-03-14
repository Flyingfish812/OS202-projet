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
    // 用于表示格子在二维数组中的坐标
    struct LexicoIndices {
        unsigned row;
        unsigned column;
    };

    // 非分布式版本构造函数（仅两进程使用）
    Model(double t_length, unsigned t_discretization, std::array<double,2> t_wind,
          LexicoIndices t_start_fire_position, double t_max_wind);

    // 分布式版本构造函数：传入 MPI 通信子、全局区域偏移以及本地 active 区域的行数
    Model(double t_length, unsigned t_discretization, std::array<double,2> t_wind,
          LexicoIndices t_start_fire_position, double t_max_wind,
          MPI_Comm t_comm, int t_global_row_offset, int t_local_rows);

    // 析构函数中释放分布式模式下复制的通信子
    ~Model();

    // 更新火势传播，返回是否仍存在燃烧区域
    bool update();

    // 获取植被和火势数据，供渲染进程使用
    const std::vector<uint8_t>& vegetal_map() const { return m_vegetation_map; }
    const std::vector<uint8_t>& fire_map() const { return m_fire_map; }

    // 返回每行的单元格数量
    unsigned geometry() const { return m_geometry; }
    // 当前时间步
    std::size_t time_step() const { return m_time_step; }

    // 全局索引映射：二维坐标 -> 一维索引（非分布式版本使用）
    std::size_t get_index_from_lexicographic_indices(LexicoIndices t_lexico_indices) const;
    // 一维索引映射：一维索引 -> 二维坐标（非分布式版本使用）
    LexicoIndices get_lexicographic_from_index(std::size_t t_global_index) const;

    // 分布式版本下：局部数组中（包含 ghost 行）的索引，local_row 取值范围 0 ~ local_rows+1
    std::size_t local_index(int local_row, int col) const;

private:
    double m_length;            // 模型区域的全长
    unsigned m_geometry;        // 每行单元格数（discretization）
    double m_distance;          // 每个格子的边长
    std::array<double,2> m_wind;// 风向风速
    double m_wind_speed;        // 风速大小
    double m_max_wind;          // 最大风速

    // 数据存储（非分布式为全局数组，分布式为局部数组，均以行优先顺序存储）
    std::vector<uint8_t> m_vegetation_map;
    std::vector<uint8_t> m_fire_map;
    
    // 用于记录火势传播的活跃区域，映射：索引 -> 当前火势值
    std::map<std::size_t, uint8_t> m_fire_front;
    
    // 参数，用于火势传播计算
    double p1;
    double p2;
    double alphaEastWest, alphaWestEast;
    double alphaSouthNorth, alphaNorthSouth;
    
    std::size_t m_time_step = 0;

    // 分布式相关标志及参数：
    // 当 distributed 为 true 时，表示当前使用分布式模式计算
    bool distributed = false;
    // active 区域（不含 ghost 行）在本地数组中拥有的行数
    int local_rows = 0;
    // 本地 active 区域在全局数组中的起始行偏移
    int global_row_offset = 0;
    // 分布式计算通信子及其相关信息
    MPI_Comm comp_comm = MPI_COMM_NULL;
    int comp_rank = 0;
    int comp_size = 0;
};

#endif // MODEL_HPP
