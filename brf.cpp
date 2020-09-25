// g++-10 brf.cpp -march=native -std=c++11 -Ofast -pthread -o brf.o

#include <fstream>
#include <iostream>
#include <sstream>
#include <bitset>
#include <set>
#include <valarray>
#include <vector>       // std::vector
#include <unordered_map>
#include <map>
#include <array>
#include <functional>

#include <queue>
#include <deque>
#include <list>

#include <iomanip>
#include <string>
#include <cstring>
#include <numeric>

#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <random>

#include <cstdint>     // for uint64_t
#include <chrono>      // for std::chrono::high_resolution_clock
#include <thread>      // for std::thread
#include <atomic>

#include <limits>
#include <cmath>

#include <iterator>
#include <utility>
#include <tuple>        // std::tuple, std::make_tuple, std::tie

#include <regex>
#include <mutex>

const std::string about =
"\n"
"/* ————————————————————————————————————————————————————————————————*\n"
" * boosted random forest                                           *\n"
" * version 0.96002                                                 *\n"
" * (c) Aleksiej Ostrowski, aleksiej.ostrowski@gmail.com, 2014-2020 *\n"
" * ————————————————————————————————————————————————————————————————*/"
"\n";

const double MAGIC_1 = -88887888.3;

const double EPS = 1e-10;
const double EPS2 = std::numeric_limits<double>::epsilon(); // 2.22045e-16
double EPS_R = 1e-10;

const double MAX_DOUBLE = std::numeric_limits<double>::max();
const double LOWEST_DOUBLE = std::numeric_limits<double>::lowest();

const std::string COMMAND_FILE = "command_file.txt";

bool SERIALIZE;

uint32_t NOISE_MISS = 0;
// 0 = nothing
// 1 = only noise
// 2 = only missing
// 3 = noise + missing

uint32_t LEAK_TEST   = 1;
uint32_t BINARY_TEST = 1;

std::mutex barrier, barrier_print, barrier_holdout, barrier_test, barrier_cases_id;

uint32_t BOOST = 0;
bool EXCLUDE_SUCCESS;

bool NO_CONTROL;
bool LOAD_BIG;

double MISSING;
size_t MAX_ROWS_TRAIN_AND_CONTROL;
double  SPLIT_FOR_CONTROL;
int32_t HAVE_ELEMENTS;

// double TRAIN_AVERAGE;
// double HOLDOUT_AVERAGE;

size_t MEAN_MODEL;
int32_t MAX_ROWS_BUF;

size_t COUNT_FROM;
size_t COUNT_TO;

bool DISCRETIO;
bool OBLIVIOUS;

size_t N_FOREST;
size_t N_FOREST_ERR;
size_t N_FOREST_ERR2;

size_t ITER_ERR;

size_t EPOCH;
size_t HIDE_COLS_EPOCH;
// size_t ITERATION;
size_t DOG;

int32_t ZEITNOT; // sec.

size_t MAX_WIDTH;
size_t MAX_LEVEL;

bool FIXED_OUTPUT;

bool DEBUG_MODE;

size_t UTILIZE_CORES;
size_t n_threads;

bool TEST_ESTIMATE;

std::string TRAIN_FILE;
std::string TEST_FILE;
std::string HOLDOUT_FILE;

std::string RESULT_FILE;
std::string FOREST_FILE;
std::string LOG_FILE;

typedef std::vector<bool> T_VEC_BIT;
typedef std::vector<double> T_VEC_D;
typedef std::vector<int32_t> T_VEC_32;
typedef std::vector<uint32_t> T_VEC_U32;
typedef std::vector<size_t> T_VEC_SIZE;

typedef std::set<int32_t> T_SET_32;
typedef std::vector<T_VEC_D> T_MAT_D;

typedef std::pair<int32_t, double> T_PAIR;
typedef std::vector<T_PAIR> T_VEC_PAIR;

typedef std::vector<T_VEC_SIZE> T_MAT_SIZE;

typedef std::pair<T_VEC_D, double> T_PRED;

T_VEC_PAIR FILTER_GROW;

double VARIANCE_RESULT;
double VARIANCE_ELEMENT;

double ALPHA;
double LAMBDA;
// double KICKDOWN;
double GAMMA;

double K1;
size_t W;

size_t MAX_BRUTE_FORCE;

// size_t TREE_MUTATE;
// size_t NODE_MUTATE;

T_MAT_D train;
T_MAT_D test;
T_MAT_D holdout;
T_MAT_D train_buffer;
T_MAT_D control;

T_MAT_D train_and_control;

T_VEC_D train_last_el;

T_VEC_D test_last_el;
T_VEC_D holdout_last_el;
T_VEC_D control_last_el;

T_VEC_32 scale;
T_VEC_32 scale_epoch;

T_MAT_SIZE MAT_ID;

std::unordered_map<size_t, size_t> hash_holdout_col, hash_test_col;
std::unordered_map<size_t, size_t> global_uses_train_lines;

T_VEC_SIZE train_and_control_id;
T_VEC_SIZE holdout_id;

T_VEC_32 shuffle_;

int32_t LOSS;
int32_t MIN_OR_MAX; // 1 or 2
double BARRIER_LOSS;
double BARRIER_CORR_VAR;
double BARRIER_CORR_RES;
double BARRIER_W;
int32_t CRITERION;

double CUT_PREDICT;

size_t HISTOGRAM;

bool IsEqual(const double a, const double b, const double EPS_ = EPS)
{
    double EPS__ = std::max( std::fabs(EPS_), EPS);
    return std::fabs(a - b) < EPS__;
}

// https://stackoverflow.com/questions/13350159/when-is-it-safe-to-compare-doubles-using-standard-in-a-map-sort

typedef std::multiset<double> T_DOUBLE_SET;
std::pair<T_DOUBLE_SET::iterator, T_DOUBLE_SET::iterator>
get_equal_range( T_DOUBLE_SET & s, double d, const double epsilon = EPS )
{
    auto lower = s.lower_bound( d - epsilon );
    auto upper = s.upper_bound( d + epsilon );
    return std::make_pair( lower, upper );
}

std::pair<T_DOUBLE_SET::const_iterator, T_DOUBLE_SET::const_iterator>
get_equal_range( T_DOUBLE_SET const & s, double d, const double epsilon = EPS )
{
    auto lower = s.lower_bound( d - epsilon );
    auto upper = s.upper_bound( d + epsilon );
    return std::make_pair( lower, upper );
}

bool in_multiset( T_DOUBLE_SET const & s, double d, const double epsilon = EPS )
{
    auto range = get_equal_range( s, d, epsilon );
    return range.first != range.second;
}


double vPearson(const T_VEC_D & v1, const T_VEC_D & v2)
{
    double mu_1 = 0.0;
    double su_1 = 0.0;
    double su_2 = 0.0;
    double su_k_1 = 0.0;
    double su_k_2 = 0.0;

    auto sz = v1.size();

    for (size_t i = 0; i < sz; i++)
    {
        auto zn1 = v1[i];
        auto zn2 = v2[i];

        mu_1 += zn1 * zn2;

        su_1 += zn1;
        su_2 += zn2;

        su_k_1 += zn1 * zn1;
        su_k_2 += zn2 * zn2;
    }

    auto numerator = sz * mu_1 - su_1 * su_2;
    auto denominator = std::sqrt( (sz * su_k_1 - su_1 * su_1) * (sz * su_k_2 - su_2 * su_2) );

    if ( IsEqual( std::fabs(denominator), 0.0) )
       return 0.0;
    else
       return numerator / denominator;
}



// http://stackoverflow.com/questions/30822729/create-ranking-for-vector-of-double

T_VEC_D vRanking(const T_VEC_D & v)
{
    T_VEC_32 w(v.size());
    std::iota(begin(w), end(w), 0);
    std::sort(begin(w), end(w),
        [&v](std::size_t i, std::size_t j) { return v[i] < v[j]; });

    T_VEC_D r(w.size());
    for (std::size_t n, i = 0; i < w.size(); i += n)
    {
        n = 1;
        while (i + n < w.size() && IsEqual(v[w[i]], v[w[i+n]])) ++n;
        for (std::size_t k = 0; k < n; ++k)
        {
            r[w[i+k]] = i + (n + 1.0) / 2.0; // average rank of n tied values
            // r[w[i+k]] = i + 1;          // min
            // r[w[i+k]] = i + n;          // max
            // r[w[i+k]] = i + k + 1;      // random order
        }
    }
    return r;
}


uint64_t pack2(const uint32_t a, const uint32_t b)
{
     uint64_t id = a;
     id <<= 32;
     id ^= b;
     return id;
}

uint32_t unpack2(const uint64_t key, const uint32_t i)
{
    if (i == 0)
        return uint32_t(key >> 32);
    else
        return uint32_t((key << 32) >> 32);
}

uint64_t seed_;

std::vector<std::mt19937_64> engines_;

size_t truly_rand(const size_t min_, const size_t max_, const int32_t p_id = 0)
{
    std::uniform_int_distribution<size_t> distribution{min_, max_};
    return distribution(engines_[p_id]);
}

double truly_rand_real(const double min_, const double max_, const int32_t p_id = 0)
{
    std::uniform_real_distribution<double> distribution{min_, max_};
    return distribution(engines_[p_id]);
}

double truly_rand_real_normal(const double min_, const double max_, const double mean_, const double stddev_, const int32_t p_id = 0)
{
    std::normal_distribution<double> distribution{mean_, stddev_};

    for (;;)
    {
         auto d = distribution(engines_[p_id]);
         if ( (d >= min_) && (d <= max_) ) return d;
    }
}

double truly_rand01(const int32_t p_id = 0)
{
    std::uniform_real_distribution<double> distribution(0, 1);
    return distribution(engines_[p_id]);
}


double rand_element(const T_VEC_D & v, const int32_t p_id = 0)
{
    return v[truly_rand(0, v.size() - 1, p_id)];
}

int32_t rand_element(const T_VEC_32 & v, const int32_t p_id = 0)
{
    return v[truly_rand(0, v.size() - 1, p_id)];
}

size_t rand_element(const T_VEC_SIZE & v, const int32_t p_id = 0)
{
    return v[truly_rand(0, v.size() - 1, p_id)];
}

int32_t rand_element(const T_SET_32 & v, const int32_t p_id = 0)
{
    auto randIt = v.begin();
    std::advance(randIt, truly_rand(0, v.size() - 1, p_id));
    return *randIt;
}

T_PAIR rand_element(const T_VEC_PAIR & v, const int32_t p_id = 0)
{
    return v[truly_rand(0, v.size() - 1, p_id)];
}

typedef union
{
    uint64_t u;
    double d;
}   U64;

uint64_t hash64(const double f)
{
    U64 temp;
    temp.d = f;
    return temp.u;
}

uint64_t cut64_bits(const uint64_t a, const uint32_t bits = 30)
{
    return (a >> bits) << bits;
}

uint64_t xor64_bits(const uint64_t a, const uint32_t b)
{
    return a ^ b;
}

// if (hash_col[xor64_bits(cut64_bits(hash64(best_threshold)), col_)]++ == 0) break;

std::string make_colored(const std::string st, const std::string color = "36", const uint8_t fl = 0)
{
    /*
     * FG_DEFAULT = 39,
     * FG_BLACK = 30,
     * FG_RED = 31,
     * FG_GREEN = 32,
     * FG_YELLOW = 33,
     * FG_BLUE = 34,
     * FG_MAGENTA = 35,
     * FG_CYAN = 36,
     * FG_LIGHT_GRAY = 37,
     * FG_DARK_GRAY = 90,
     * FG_LIGHT_RED = 91,
     * FG_LIGHT_GREEN = 92,
     * FG_LIGHT_YELLOW = 93,
     * FG_LIGHT_BLUE = 94,
     * FG_LIGHT_MAGENTA = 95,
     * FG_LIGHT_CYAN = 96,
     * FG_WHITE = 97,

     * BG_RED = 41,
     * BG_GREEN = 42,
     * BG_BLUE = 44,
     * BG_DEFAULT = 49
     */
    auto a1 = "\033[1;" + color + "m";
    auto a2 = "\033[0m";
    switch (fl)
    {
        case 1  : return a1 + st;
        case 2  : return st + a2;
        default : return a1 + st + a2;
    }
}

std::string print_sec(const int32_t s, const bool no_color = false)
{
    auto num_seconds = s;
    auto days = num_seconds / (60 * 60 * 24);
    num_seconds -= days * (60 * 60 * 24);

    auto hours = num_seconds / (60 * 60);
    num_seconds -= hours * (60 * 60);

    auto minutes = num_seconds / 60;
    num_seconds -= minutes * 60;

    std::string w1 = make_colored("", "97", 1);
    std::string w2 = make_colored("", "97", 2);

    if (no_color)
    {
        w1 = ""; w2 = "";
    }

    std::ostringstream ss;

    if (days > 0) ss << w1;
    ss << days << " d ";
    if (days > 0) ss << w2;

    if (hours > 0) ss << w1;
    ss << hours << " h ";
    if (hours > 0) ss << w2;

    if (minutes > 0) ss << w1;
    ss << minutes << " m ";
    if (minutes > 0) ss << w2;

    if (num_seconds > 0) ss << w1;
    ss << num_seconds << " s ";
    if (num_seconds > 0) ss << w2;

    return ss.str();
}

std::string print_now()
{
    using std::chrono::system_clock;
    std::time_t tt = system_clock::to_time_t (system_clock::now());
    auto p = std::localtime(&tt);

    std::ostringstream ss;
    ss << std::put_time(p, "%F %T");

    return ss.str();
}

std::string print_loss(const double l)
{
    if (IsEqual(l, MAGIC_1))
        return "nan";
    else
        return std::to_string(l);
}

size_t MYHash(const std::string & str)
{
    std::hash<std::string> hash_my;
    return hash_my(str);
}


// http://stackoverflow.com/questions/236129/split-a-string-in-c

T_VEC_32 &split32(const std::string &s, const char delim, T_VEC_32 & elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        elems.push_back(std::stoi(item));
    return elems;
}


T_VEC_32 split32(const std::string & s, const char delim)
{
    T_VEC_32 elems;
    split32(s, delim, elems);
    return elems;
}

T_VEC_U32 &splitU32(const std::string & s, const char delim, T_VEC_U32 & elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        elems.push_back(std::stoul(item));
    return elems;
}


T_VEC_U32 splitU32(const std::string & s, const char delim)
{
    T_VEC_U32 elems;
    splitU32(s, delim, elems);
    return elems;
}

void print_value(const double & d, const std::string & s)
{
    std::lock_guard<std::mutex> lock(barrier_print);
    std::cout << s << std::endl;
    std::cout << d << std::endl;
    std::cout << "------------- " << std::endl;
}

void print_vector(const T_VEC_D & v)
{
    std::lock_guard<std::mutex> lock(barrier_print);
    std::cout << "print vector: " << std::endl;
    for(auto el : v) std::cout << el << " ";
    std::cout << std::endl << "------------- " << std::endl;
}

void print_vector(const T_VEC_BIT & v)
{
    std::lock_guard<std::mutex> lock(barrier_print);
    std::cout << "print vector: " << std::endl;
    for(auto el : v) std::cout << el << " ";
    std::cout << std::endl << "------------- " << std::endl;
}

void print_vector(const T_VEC_32 & v)
{
    std::lock_guard<std::mutex> lock(barrier_print);
    std::cout << "print vector: " << std::endl;
    for(auto el : v) std::cout << el << " ";
    std::cout << std::endl << "------------- " << std::endl;
}


void print_log(const std::string & s)
{
    std::lock_guard<std::mutex> lock(barrier_print);
    std::cout << s << std::endl << "------------- " << std::endl;
}


T_VEC_D &splitD(const std::string & s, const char delim, T_VEC_D & elems, const bool is_scale = false)
{
    std::stringstream ss(s);
    std::string item;
    size_t i = 0;
    while (std::getline(ss, item, delim))
    {
        double el;
        if ( is_scale && (i <= scale.size() - 1) )
        {
            switch (scale[i++])
            {
                case 1:
                {
                    try
                    {
                        el = std::stod(item);
                    }
                    catch (...)
                    {
                        el = MYHash(item);
                    }
                    break;
                }
                default: el = std::stod(item); break;
            }
        }
        else
            el = std::stod(item);

        elems.push_back(el);
    }
    return elems;
}


T_VEC_D splitD(const std::string & s, const char delim, const bool is_scale = false)
{
    T_VEC_D elems;
    splitD(s, delim, elems, is_scale);
    return elems;
}


T_VEC_PAIR &splitPAIR(const std::string & s, const char delim, T_VEC_PAIR & elems)
{
    // const std::string st = "12:23;4:6-10;2:2-3";
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        std::string key, val;
        std::stringstream s_item(item);
        std::getline(s_item, key, ':');
        std::getline(s_item, val);
        auto key_ = std::stoi(key);

        std::string from_, to_;
        std::stringstream s_subitem(val);
        std::getline(s_subitem, from_, '-');
        std::getline(s_subitem, to_);

        if ( from_.empty() || to_.empty() )
        {
            auto v = std::stod(val);
            elems.push_back(std::make_pair(key_, v));
        }
        else
        {
            auto from__ = std::stoi(from_);
            auto to__ = std::stoi(to_);

            for (auto i = from__; i <= to__; i++)
            {
                elems.push_back(std::make_pair(key_, (double)i));
            }
        }
    }
    return elems;
}


T_VEC_PAIR splitPAIR(const std::string & s, const char delim = ';')
{
    T_VEC_PAIR elems;
    splitPAIR(s, delim, elems);
    return elems;
}


T_VEC_PAIR &splitPAIR_DOUBLE(const std::string & s, char delim, T_VEC_PAIR & elems)
{
    // const std::string st = "1-10:0.5;11:0.6;12-20:1.2";

    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        std::string key, val;
        std::stringstream s_item(item);
        std::getline(s_item, key, ':');
        std::getline(s_item, val);

        auto val_ = std::stod(val);

        std::stringstream s_subitem(key);
        std::string from_, to_;
        std::getline(s_subitem, from_, '-');
        std::getline(s_subitem, to_);

        if ( from_.empty() || to_.empty() )
        {
            auto key_ = std::stoi(key);
            elems.push_back(std::make_pair(key_, val_));
        }
        else
        {
            auto from__ = std::stoi(from_);
            auto to__ = std::stoi(to_);

            for (auto i = from__; i <= to__; i++)
            {
                elems.push_back(std::make_pair(i, val_));
            }
        }
    }
    return elems;
}


T_VEC_PAIR splitPAIR_DOUBLE(const std::string & s, char delim = ';')
{
    T_VEC_PAIR elems;
    splitPAIR_DOUBLE(s, delim, elems);
    return elems;
}

size_t load_command(const std::string fl)
{
    size_t r = 0;
    std::ifstream myfile;
    myfile.open(fl);

    if (myfile.is_open())
    {
        std::string line = "";
        if (std::getline(myfile, line))
        {
            if (line == "stop")
            {
                r = 1;
                std::cout << std::endl << make_colored("", "31", 1) << "STOP WORK." << make_colored("", "31", 2) << std::endl << std::endl;
            }
        }
    }

    myfile.close();

    return r;
}

void load_test(const std::string fl)
{
    std::ifstream myfile;
    std::string ftest = fl;

    for(;;)
    {
        myfile.open(ftest);
        if (myfile.is_open()) break;
        myfile.clear();
        std::cout << "unknown test file: " << ftest << std::endl << "please, input new name of TEST file (# - exit): ";
        std::getline(std::cin, ftest);
        if (ftest == "#") exit(1);
    }

    test.clear();
    std::string line = "";
    while(std::getline(myfile, line))
        test.push_back(splitD(line, ',', true));

    myfile.close();

    if (TEST_ESTIMATE)
    {
        test_last_el.clear();
        for (size_t row = 0; row < test.size(); row++)
            test_last_el.push_back(test[row].back());
    }

    if (DEBUG_MODE)
    {
        size_t i = 0;
        for (auto & e : test)
        {
            std::cout << "test[" << i++ << "]: ";
            for (auto & x : e)
                 std::cout << x << " ";
            std::cout << std::endl;
        }
    }
}

void knuth_shuffle_train_and_control(const int32_t p_id = 0)
{
    auto sz_y = train_and_control.size();
    auto sz_x = train_and_control[0].size();

    for (size_t col = 0; col < sz_x; col++)
    {
        if (shuffle_[col] == 0) continue;

        for (size_t z = 0; z < sz_y; z++)
        {
            auto k = truly_rand(0, sz_y, p_id);
            if (k != z)
                std::swap(train_and_control[k][col], train_and_control[z][col]);
        }
    }
}

template <typename KX>
KX knuth_shuffle(const KX & v, const int32_t p_id = 0)
{
    KX r(v);

    auto sz_r = r.size();
    for (size_t z = 0; z < sz_r; z++)
    {
        auto k = truly_rand(0, sz_r, p_id);
        if (k != z) std::swap(r[k], r[z]);
    }

    return r;
}

void load_train_and_control(const std::string fl)
{
    std::ifstream myfile;
    std::string ftrain_and_control = fl;

    for(;;)
    {
        myfile.open(ftrain_and_control);
        if (myfile.is_open()) break;
        myfile.clear();
        std::cout << "unknown train_and_control file: " << ftrain_and_control << std::endl << "please, input new name of TRAIN_AND_CONTROL file (# - exit): ";
        getline(std::cin, ftrain_and_control);
        if (ftrain_and_control == "#") exit(1);
    }

    train_and_control.clear();
    train_and_control_id.clear();

    std::string line = "";
    while (std::getline(myfile, line))
    {
        train_and_control.push_back(splitD(line, ',', true));
        train_and_control_id.push_back(MYHash(line));
    }

    knuth_shuffle_train_and_control();

    myfile.close();
}


void load_holdout(const std::string fl)
{
    std::ifstream myfile;
    std::string fval = fl;

    for(;;)
    {
        myfile.open(fval);
        if (myfile.is_open()) break;
        myfile.clear();
        std::cout << "unknown holdout file: " << fval << std::endl << "please, input new name of holdout file (# - exit): ";
        std::getline(std::cin, fval);
        if (fval == "#") exit(1);
    }

    holdout.clear();
    holdout_id.clear();

    std::string line = "";
    while (std::getline(myfile, line))
    { 
        holdout.push_back(splitD(line, ',', true));
        holdout_id.push_back(MYHash(line));
    }

    myfile.close();

    holdout_last_el.clear();
    for (size_t row = 0; row < holdout.size(); row++)
        holdout_last_el.push_back(holdout[row].back());

    // double sum = std::accumulate(std::begin(holdout_last_el), std::end(holdout_last_el), 0.0);
    // HOLDOUT_AVERAGE = sum / (double) holdout_last_el.size();

    if (DEBUG_MODE)
    {
        size_t i = 0;
        for (auto & e : holdout)
        {
            std::cout << "holdout[" << i++ << "]: ";
            for (auto & x : e)
                 std::cout << x << " ";
            std::cout << std::endl;
        }
    }
}


std::string load_config(const std::string fl)
{
    std::ifstream myfile(fl, std::ifstream::in);
    if (!myfile)
    {
        std::cout << "unknown config file: " << fl << std::endl;
        exit(1);
    }
    std::string r = "", line = "";
    while(std::getline(myfile, line))
        r += line + "\n";
    myfile.close();
    return r;
}

std::string template_parsing(const std::string s)
{
    std::string s_new(s);

    int32_t start_ = 0;

    std::regex r("\".*?\"\\*[0-9]{1,}");

    for (auto i = std::sregex_iterator(s.begin(), s.end(), r);i != std::sregex_iterator();++i)
    {
        std::smatch m = *i;
        std::string f = m.str();

        auto df = f.length();
        auto d1 = f.find("\"*");
        auto s1 = f.substr(1, d1 - 1);
        auto s2 = f.substr(d1 + 2, df - d1 - 2);

        std::string tmp = "";
        size_t n = stoi(s2);

        for (size_t j = 0; j < n; ++j)
            tmp += s1;

        auto t_= m.position() + start_;

        s_new.erase(t_, df);
        s_new.insert(t_, tmp);

        start_ += tmp.length() - df;
    }
    return s_new;
}

std::string read_tag(const std::string tag, const std::string st, const std::string default_ = "")
{
    auto st_ = " " + st;
    auto tt = tag + ">";
    auto t1 = "<" + tt;
    auto t2 = "</" + tt;
    auto d1 = st_.find(t1);

    if (d1 == std::string::npos)
        return template_parsing(default_);
    else
    {
        auto ds1 = d1 + t1.length();
        auto d2 = st_.find(t2, d1);
        if (d1 == std::string::npos)
            return template_parsing(default_);
        else
            return template_parsing(st_.substr(ds1, d2 - ds1));
    }
}

int32_t str_to_int(const std::string st, const int32_t default_ = 0)
{
    try
    {
        return std::stoi(st);
    }
    catch(...)
    {
        return default_;
    }
}

double str_to_double(const std::string st, const double default_ = 0.0)
{
    try
    {
        return std::stod(st);
    }
    catch(...)
    {
        return default_;
    }
}

bool str_to_bool(const std::string st, const bool default_ = false)
{
    if (st.empty())
       return default_;
    else
       return st == "true";
}


void print_train_and_control()
{
    std::cout << "train: -------------------------------" << std::endl;

    size_t i = 0;
    for (auto &e : train)
    {
        std::cout << "train[" << i++ << "]: ";
        for (auto &x : e)
             std::cout << x << " ";
        std::cout << std::endl;
    }

    if (!NO_CONTROL)
    {
        std::cout << "control: -------------------------------" << std::endl;

        i = 0;
        for (auto &e : control)
        {
            std::cout << "control[" << i++ << "]: ";
            for (auto &x : e)
                 std::cout << x << " ";
            std::cout << std::endl;
        }
    }

    std::cout << "----------------------------------------" << std::endl;

}

/*
double jaccard(const T_VEC_D & a, const T_VEC_D & b)
{
    double c = intersect_double_vector(a, b).size();
    return c / ( a.size() + b.size() - c );
}
*/


T_VEC_D sum_vec_d(const T_VEC_D & a, const T_VEC_D & b, const double c = 1.0)
{
    T_VEC_D r(a.size(), MAGIC_1);

    for (size_t i = 0; i < r.size(); i++)
    {
        if ( !IsEqual(a[i], MAGIC_1) && !IsEqual(b[i], MAGIC_1) )
            r[i] = a[i] + c * b[i];
    }

    return r;
}


T_VEC_D sub_vec_d(const T_VEC_D & a, const T_VEC_D & b, const double c = 1.0)
{
    T_VEC_D r(a.size(), MAGIC_1);

    for (size_t i = 0; i < r.size(); i++)
    {
        if ( !IsEqual(a[i], MAGIC_1) && !IsEqual(b[i], MAGIC_1) )
            r[i] = a[i] - c * b[i];
    }

    return r;
}

T_VEC_D mul_vec_d(const T_VEC_D & a, const double c = 1.0)
{
    T_VEC_D r(a.size(), MAGIC_1);

    for (size_t i = 0; i < r.size(); i++)
    {
        if ( !IsEqual(a[i], MAGIC_1) )
            r[i] = c * a[i];
    }

    return r;
}


double counter_(const T_VEC_D & v, const double e, const double EPS_ = EPS)
{
    double EPS__ = std::max( std::fabs(EPS_), EPS);
    double sum = 0.0;
    for (const auto & x : v)
    {
        if (IsEqual(x, e, EPS__)) sum += 1.0;
    }
    return sum;
}

// http://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
std::pair<double, double> average_variance(const T_VEC_D & v)
{
    switch (v.size())
    {
        case 0: return std::make_pair(MAGIC_1, 0.0);
        case 1: return std::make_pair(v[0], 0.0);
        default:
        {
            double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
            double m = sum / (double) v.size();
            double accum = 0.0;

            std::for_each (std::begin(v), std::end(v), [&](const double d) { accum += (d - m) * (d - m); });

            return std::make_pair(m, accum / (v.size() - 1.0)); // stdev sqrt(...)
        }
    }
}


std::size_t split_train_and_control()
{
    auto sz = train_and_control.size();
    std::size_t const div2 = std::round(sz * SPLIT_FOR_CONTROL);

    control.resize(div2);

    auto end_ch = sz - div2;

    train.resize(end_ch);

    std::copy(train_and_control.begin(),           train_and_control.begin() + div2, control.begin());
    std::copy(train_and_control.begin()    + div2, train_and_control.end(),          train.begin());

    train_last_el.resize(end_ch);

    for (size_t row = 0; row < end_ch; row++)
        train_last_el[row] = train[row].back();

    // double sum = std::accumulate(std::begin(train_last_el), std::end(train_last_el), 0.0);
    // TRAIN_AVERAGE = sum / (double) train_last_el.size();

    control_last_el.resize(div2);
    for (size_t row = 0; row < div2; row++)
        control_last_el[row] = control[row].back();

    if (DEBUG_MODE)
        print_train_and_control();

    return div2;
}


void load_big_train_and_control(const std::string fl, std::unordered_map<std::string, uint32_t> & map_stop_line)
{
    std::ifstream myfile(fl, std::ifstream::in);
    if (!myfile)
    {
        std::cout << "unknown train big file: " << fl << std::endl;
        exit(1);
    }

    myfile.seekg(0, std::ios::end);
    size_t filesize_ = myfile.tellg();

    if (DEBUG_MODE)
        std::cout << "SIZE BIG :: " << filesize_ << std::endl;

    T_VEC_D vec_last_el;
    vec_last_el.reserve(MAX_ROWS_TRAIN_AND_CONTROL);

    train_and_control.clear();
    train_and_control_id.clear();

    if (HAVE_ELEMENTS != 0)
    {
        train_and_control.reserve(MAX_ROWS_TRAIN_AND_CONTROL);
        train_and_control_id.reserve(MAX_ROWS_TRAIN_AND_CONTROL);
    }

    size_t j = 0;

    auto aa = std::chrono::high_resolution_clock::now();

    for (;;)
    {
        myfile.seekg(truly_rand(0, filesize_ - 1), std::ios::beg);
        myfile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        std::string line = "";
        std::getline(myfile, line);

        if (myfile.eof() || myfile.bad() || myfile.fail())
            myfile.clear();

        if (line.length() < 1) continue;

        auto bb = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::duration_cast<std::chrono::duration<double>>(bb - aa).count();

        if (t1 > ZEITNOT)
        {
            aa = bb;
            map_stop_line.clear();
        }

        if (map_stop_line[line]++ > 0) continue;

        auto temp = splitD(line, ',', true);

        if (HAVE_ELEMENTS != 0)
        {
            auto el = temp.back();
            auto cnt = counter_(vec_last_el, el, el * EPS_R);
            if ( cnt > HAVE_ELEMENTS ) continue;
            vec_last_el.push_back(el);

            if (DEBUG_MODE)
            {
                std::cout << "training result = " << el << " n = " << cnt << " at line = " << j << "/" << MAX_ROWS_TRAIN_AND_CONTROL << std::endl;
            }
        }

        train_and_control.push_back(temp);
        train_and_control_id.push_back(MYHash(line));

        if (DEBUG_MODE)
        {
            auto sz = line.length();
            line.resize(40, '.');
            std::cout << "BIG :: " << "\"" << line << "\"" << "  " << sz << std::endl;
        }

        if (j++ > MAX_ROWS_TRAIN_AND_CONTROL) break;
    }

    knuth_shuffle_train_and_control();

    myfile.close();
}

struct resTree
{
    bool flag = false;
    double res = 0.0;

    size_t id1 = 0;
    size_t id2 = 0;

    double w = 0.0;
};

struct BinaryTree
{
    size_t id = 0;
    size_t level = 0;

    int32_t col = -1;

    double threshold = 0.0;
    double res = MAGIC_1;

    double early = 1.0;
    double opt = MAGIC_1;

    uint32_t size_res = 0;

    double err = 0.0;

    uint32_t flag = 3;

    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t parent = 0;
};

typedef std::vector<BinaryTree> T_TREE;

typedef std::vector<T_TREE> T_FOREST;
typedef std::vector<T_FOREST> M_FOREST;

std::string open_tag(const std::string name_, const size_t bs = 0)
{
    std::ostringstream s;
    for (size_t i = 0; i < bs; i++) s << " ";
    s << "<" << name_ << ">";
    return s.str();
}

std::string close_tag(const std::string name_, const size_t bs = 0)
{
    std::ostringstream s;
    for (size_t i = 0; i < bs; i++) s << " ";
    s << "</" << name_ << ">";
    return s.str();
}

std::string vec_to_str(const T_VEC_D & a)
{
    std::ostringstream s;
    for (auto x : a) s << x << ",";
    auto s_new = s.str();
    return s_new.substr(0, s_new.size() - 1);
}


template <typename X>
std::string wrap_(X x, const std::string name_, const int32_t bs = 0)
{
    std::ostringstream s;
    s << open_tag(name_, bs) << x << close_tag(name_) << std::endl;
    return s.str();
}

void serialize(const T_FOREST & F, std::ofstream & os, const std::string tag = "")
{
    for (size_t y = 0; y < F.size(); y++)
    {
        auto T = F[y];

        if (y == 1) os << "/*" << std::endl;

        os << "digraph \"FOREST_" << tag << "_TREE_" << y + 1 << "\" {" << std::endl;

        os << "    0 [shape=box, style=filled, color=hotpink, label=\"ROOT_" << tag << "_" << y + 1 <<"\"];" << std::endl;

        for (size_t i = 1; i < T.size(); i++)
        {
            auto el = T[i];
            if (el.flag != 3)
            {
                auto is_left = T[el.parent].left == i;

                os << "    " << el.parent << " -> " << i << ";" << std::endl;

                if (el.flag == 1)
                    // os << "    " << i << " [shape=box, label=\"V" << el.col + 1;
                    os << "    " << i << " [shape=record, style=filled, color=khaki, label=\"{V" << el.col + 1;

                if (el.flag == 2)
                    os << "    " << i << " [shape=record, style=filled, color=skyblue, label=\"{V" << el.col + 1;

                if (is_left)
                    os << " \\< ";
                else
                    os << " \\>= ";

                os << el.threshold;

                // if (el.flag == 2)
                os << "| res = " << el.res << "|{level = " << el.level << "}}";

                os << "\"];" << std::endl;

           }
        }

        os << "}" << std::endl << std::endl;
    }

   if (F.size() > 1) os << "*/" << std::endl;
}


bool delete_trash(T_TREE & tree, const size_t t = 0)
{
    auto f1 = (tree[t].left == 0);
    auto f2 = (tree[t].right == 0);

    if ( f1 && f2 )
        return false;

    bool a1 = false, a2 = false;
    bool res1 = false, res2 = false;

    if (!f1)
    {
        if (tree[tree[t].left].flag == 3)
        {
            tree[t].left = 0;
            a1 = true;
            res1 = true;
        }
        else
            res1 = delete_trash(tree, tree[t].left);
    }

    if (!f2)
    {
        if (tree[tree[t].right].flag == 3)
        {
            tree[t].right = 0;
            a2 = true;
            res2 = true;
        }
        else
            res2 = delete_trash(tree, tree[t].right);
    }

    bool res3 = false;

    if ( ( f1 && a2 ) || ( f2 && a1 ) || ( a1 && a2) )
    {
        tree[t].flag = 3;
        res3 = true;
    }

    return res1 || res2 || res3;

}

/*

void cut_tree(T_TREE & tree)
{
    for (size_t t = 0; t < tree.size(); t++)
    {
        if ( (tree[t].flag == 1) || (tree[t].flag == 2))
        {
            if ( (IsEqual(tree[tree[t].parent].res, tree[t].res, EPS_R)) )
            {
                tree[tree[t].parent].flag = 3;
                // tree[tree[t].parent].left = 0;
                // tree[tree[t].parent].right = 0;
                // tree[t].flag = 3;
            }
        }
    }
}

*/

// see ... https://github.com/sol-prog/threads/blob/master/part_2/cpp11_threads_03.cpp
T_VEC_32 bounds(const int32_t parts, const int32_t mem)
{
    T_VEC_32 bnd;
    auto delta = mem / parts;
    auto reminder = mem % parts;
    int32_t N1 = 0, N2 = 0;
    bnd.push_back(N1);
    for (int32_t i = 0; i < parts; ++i)
    {
        N2 = N1 + delta;
        if (i == parts - 1)
            N2 += reminder;
        bnd.push_back(N2);
        N1 = N2;
    }
    return bnd;
}

// Воронцов К.В. Лекции по логическим алгоритмам классификации. 24 июня 2010 г. C.5
double rama(const double k)
{
    return k * std::log(k) - k + std::log(k * (1.0 + 4.0 * k * (1.0 + 2.0 * k))) * 0.16666666666 + 0.5 * std::log(M_PI);
}

double logCkn(const double k, const double n)
{
    return rama(n) - rama(k) - rama(n-k);
}

double Ic(const double p, const double n, const double P, const double N)
{
    return -logCkn(p, P) - logCkn(n, N) + logCkn(p + n, P + N);
    // >= − log(0.05)
}


void make_permit(const size_t a1, const size_t a2, const T_VEC_32 & ia, const T_VEC_32 & ib, const T_MAT_D & mat, T_VEC_D & permit_p)
{
    const size_t n_len = mat.size();

    for (size_t i = a1; i < a2; i++)
    {
        const auto idx_from_ia = ia[i];
        const auto idx_from_ib = ib[i];

        T_VEC_D a(n_len);
        T_VEC_D b(n_len);

        for (size_t j = 0; j < n_len; j++)
        {
            a[j] = mat[j][idx_from_ia];
            b[j] = mat[j][idx_from_ib];
        }

        T_VEC_D a_, b_;

        a_.reserve(n_len);
        b_.reserve(n_len);

        for (size_t j = 0; j < n_len; j++)
        {
            if (
                  !IsEqual(a[j], MAGIC_1) && !IsEqual(b[j], MAGIC_1) &&
                  !IsEqual(a[j], MISSING) && !IsEqual(b[j], MISSING)
               )
            {
                a_.push_back(a[j]);
                b_.push_back(b[j]);
            }
        }

        permit_p[i] = vPearson(a_, b_);
    }
}


T_VEC_32 prepare_scale(const T_VEC_32 scale_const)
{
    const auto sz_row = train.size();
    const auto sz_col = train[0].size();

    T_VEC_32 r(scale_const);

    T_MAT_D new_train;
    new_train.resize(sz_col);

    for (size_t i = 0; i < sz_col; i++)
    {
        new_train[i].resize(sz_row);
        for (size_t j = 0; j < sz_row; j++)
            new_train[i][j] = train[j][i];
    }

    T_VEC_SIZE super;

    for (size_t col = 0; col < sz_col; col++)
    {
        if (r[col] != 0) continue;

        T_VEC_D v0;
        v0.reserve(sz_row);

        T_VEC_D rr;
        rr.reserve(sz_row);

        for (size_t row = 0; row < sz_row; row++)
            if (!IsEqual(new_train[col][row], MISSING) && !IsEqual(train_last_el[row], MAGIC_1))
            {
                v0.push_back(new_train[col][row]);
                rr.push_back(train_last_el[row]);
            }  

        if (std::fabs(vPearson(v0, rr)) > BARRIER_CORR_RES) super.push_back(col);

        auto a_pair = average_variance(v0);

        if (a_pair.second < VARIANCE_ELEMENT)
            r[col] = 2;
    }

    // for (size_t i = 0; i < sz_col; i++)
    //     new_train[i] = vRanking(new_train[i]);

    const size_t e_len = sz_col;

    T_VEC_32 new_index = {0};

    if (e_len > 1)
    {
        T_VEC_32 ia, ib;

        for (size_t i = 0; i < e_len; i++)
        {
            if ( r[i] != 0) continue;  
            for (size_t j = i + 1; j < e_len; j++)
            {
                 if (r[j] != 0) continue;  
                 ia.push_back(i);
                 ib.push_back(j);
            }
        }

        const size_t ab_len = ia.size();

        if (ab_len > 2)
        {
 
            T_VEC_D permit_p(ab_len);

            {

            size_t n_threads_ = n_threads;

            if (n_threads > ab_len)
                n_threads_ = ab_len;

            T_VEC_32 limits = bounds(n_threads_, ab_len);

            std::vector<std::thread> th;
            for (size_t id_ = 0; id_ < n_threads_; id_++)
                th.push_back(std::thread(make_permit, limits[id_], limits[id_ + 1] - 1, std::ref(ia), std::ref(ib), std::ref(train), std::ref(permit_p)));

            for (auto &t : th) t.join();

            }

            T_VEC_D permit_(e_len, 0.0);

            for (size_t i = 0; i < ab_len; i++)
            {
                const auto idx_from_ia = ia[i];
                const auto idx_from_ib = ib[i];

                if ( (r[idx_from_ia] == 2) || (r[idx_from_ib] == 2) ) continue;

                if (std::fabs(permit_p[i]) > BARRIER_CORR_VAR)
                {
                    /*
                    if ( (idx_from_ia == 1) || (idx_from_ib == 1) )
                    {
                        std::stringstream ss;
                        ss << idx_from_ia << " - " << idx_from_ib;
                        std::string s = ss.str();

                        print_log(s);
                    }
                    */ 

                    if (truly_rand01() > 0.5)
                        r[idx_from_ia] = 2;
                    else
                        r[idx_from_ib] = 2;
                }
            }

            /*
            if ((super.size() > 0) && (truly_rand01() > 0.5))
            for (size_t i = 0; i < e_len; i++)
                if (std::find(super.begin(), super.end(), i) == super.end()) r[i] = 2; else r[i] = 0;
            */
        }
    }

    for (size_t i = 0; i < HIDE_COLS_EPOCH; i++)
       r[truly_rand(0, sz_col)] = 2;

    return r;
}



double max_abs(const T_VEC_D & v)
{
    double r = std::fabs(v[0]);
    for (auto & x : v)
        r = std::max(std::fabs(x), r);
    return r;
}

double entropy(const T_VEC_D & v)
{
    double e = 0.0;
    for (auto x : v)
        if (!IsEqual(x, 0.0))
            e += std::log(std::fabs(x));

    return ( - e / (double) v.size() );
}


double median(const T_VEC_D & data)
{
    T_VEC_D new_data;
    new_data.reserve(data.size());

    for (auto & x : data)
        if (!IsEqual(x, MAGIC_1)) new_data.push_back(x);

    if (new_data.size() == 0) return MAGIC_1;

    if (new_data.size() == 1) return new_data[0];

    size_t n = new_data.size() / 2;
    std::nth_element( new_data.begin(), new_data.begin() + n, new_data.end() );
    return new_data[n];
}

// http://stackoverflow.com/questions/19980319/efficient-way-to-compute-geometric-mean-of-many-numbers

double geometric_mean(const T_VEC_D & data)
{
    T_VEC_D new_data;
    new_data.reserve(data.size());

    for (auto & x : data)
        if (!IsEqual(x, MAGIC_1)) new_data.push_back(x);

    if (new_data.size() == 0) return MAGIC_1;

    if (new_data.size() == 1) return new_data[0];

    double m = 1.0;
    long long ex = 0;
    double invN = 1.0 / (double) new_data.size();

    for (auto & x : new_data)
    {
        int i;
        double f1 = std::frexp(x, &i);
        m  *= f1;
        ex += i;
    }

    return std::pow( std::numeric_limits<double>::radix, ex * invN) * std::pow(m, invN);
}

double res_r(const T_VEC_D & v)
{
    auto sz = v.size();
    if (sz == 0) return MAGIC_1;

    double sum_d = 0.0;
    size_t nn = 0;

    for (const auto & x : v)
        if (!IsEqual(x, MAGIC_1))
        {
            sum_d += x;
            nn++;
        }

    if (nn == 0) return MAGIC_1;

    return sum_d / (double) nn;
}

T_VEC_D my_clear_vector(const T_VEC_D & v)
{
    auto l = v.size();
    if (l <= 7) return v;

    T_VEC_D v_(v);
    std::sort(v_.begin(), v_.end());

    size_t i1 = round(0.25 * l);
    size_t i2 = round(0.75 * l);

    auto first = v_.begin() + i1;
    auto last  = v_.begin() + i2;

    T_VEC_D res_(first, last);

    return res_;
}

double shoorygin(const T_VEC_D & vec)
{
    T_VEC_D v(vec);
    if (v.empty()) return MAGIC_1;

    std::sort(v.begin(), v.end());

    for (;;)
    {
        auto l = v.size();
        if (l <= 2)
        {
            double sum_ = 0.0;
            for (const auto x : v)  sum_ += x;
            return sum_ / l;
        }

        auto b = (v.front() + v.back()) * 0.5;
        if (b <= v[1])
        {
            v.erase(v.begin());
            continue;
        }

        if (b >= v[v.size() - 2])
        {
            v.pop_back();
            continue;
        }

        v.erase(v.begin());
        v.pop_back();

        v.insert(upper_bound(v.begin(),v.end(), b), b);
    }
}

/*

T_VEC_D err_normalize(const T_VEC_D & v)
{
    T_VEC_D v_(v);

    // double m_ = LOWEST_DOUBLE;

    for (auto & x : v_)
        x = std::fabs(x);

    // auto sh = 0.97 * m_;

    auto sh = shoorygin(v_);

    v_ = v;

    for (auto & x : v_)
        if (std::fabs(x) > sh) x = 0.0; // MAGIC_1;

    return v_;
}
*/

T_VEC_D concat(const T_VEC_D & a, const T_VEC_D  & b)
{
    T_VEC_D r(a);
    r.insert(r.end(), b.begin(), b.end());
    return r;
}


bool is_compress(const T_VEC_D & a, const T_VEC_D & b)
{
    T_VEC_D a_(a);
    T_VEC_D b_(b);

    auto len_a = a.size();
    auto len_b = b.size();

    // if (len_a + len_b > 10000) return true;

    std::sort(a_.begin(), a_.end());
    a_.erase(std::unique (a_.begin(), a_.end(), [](double x1, double x2) { return IsEqual(x1, x2, (x1 + x2) * 0.5 * EPS_R); } ), a_.end());

    std::sort(b_.begin(), b_.end());
    b_.erase(std::unique (b_.begin(), b_.end(), [](double x1, double x2) { return IsEqual(x1, x2, (x1 + x2) * 0.5 * EPS_R); } ), b_.end());

    auto c_ = concat(a, b);
    auto len_c = c_.size();

    std::sort(c_.begin(), c_.end());
    c_.erase(std::unique (c_.begin(), c_.end(), [](double x1, double x2) { return IsEqual(x1, x2, (x1 + x2) * 0.5 * EPS_R); } ), c_.end());

    double k_a_ = a_.size() / (double) len_a;
    double k_b_ = b_.size() / (double) len_b;
    double k_c_ = c_.size() / (double) len_c;

    /*
    std::cout << "k_a_ = " << k_a_ << std::endl;
    std::cout << "k_b_ = " << k_b_ << std::endl;
    std::cout << "k_c_ = " << k_c_ << std::endl;
    */

    bool r = (k_a_ < k_c_) or (k_b_ < k_c_);

    if (not r)
    {
        r = IsEqual(k_a_, k_c_) and IsEqual(k_b_, k_c_);
    }

    return r;
}


std::pair<double, double> median_median(const std::vector<double> & v)
{
    auto vs = v.size();
    switch (vs)
    {
        case 0 : return std::make_pair(MAGIC_1, MAGIC_1);
        case 1 : return std::make_pair(v[0], v[0]);
        default:
        {
            std::vector<double> v_(v);
            size_t n = vs / 2;
            std::nth_element( v_.begin(), v_.begin() + n, v_.end() );

            std::vector<double> a (v_.begin(), v_.begin() + n);
            std::vector<double> b (v_.begin() + n, v_.end());

            size_t n_a = a.size() / 2;
            std::nth_element( a.begin(), a.begin() + n_a, a.end() );
            double r1 = a[n_a];

            size_t n_b = b.size() / 2;
            std::nth_element( b.begin(), b.begin() + n_b, b.end() );
            double r2 = b[n_b];

            return std::make_pair(r1, r2);
        }
    }
}

void endlich(BinaryTree & el, const T_VEC_BIT & r, const T_VEC_SIZE & train_last_id, const bool uses_train_id_lines = false)
{
    if (el.flag == 1)
    {
        el.flag = 2;
        el.early = 0.1;

        if (uses_train_id_lines)
        {
            double ww = 1.0;

            for (size_t i = 0; i < r.size(); i++)
                if (r[i])
                {
                    std::lock_guard<std::mutex> lock(barrier_cases_id);
                    ww += (double) global_uses_train_lines[train_last_id[i]];
                }

            ww = std::log(ww);
            el.early = 1.0 / ( ww * ww * 2.0 );
        }
    }
}

double confidence(const size_t ups, const size_t downs)
{
    auto n = ups + downs;
    if (n == 0)  return 0.0;
    double z = 1.2; // 1.0 = 85%, 1.6 = 95%
    double phat = (double) ups / n;
    return std::sqrt(phat + z * z / (2.0 * n) - z * (( phat * (1.0 - phat) + z * z / (4.0 * n)) / n)) / (1.0 + z * z / n);
}

double correction(const size_t n, const double k1 = K1)
{
    return std::sqrt((double) n / ((double) n + k1));
}

T_PAIR moda(const T_VEC_D & v)
{
    auto v_(v);

    std::sort(std::begin(v_), std::end(v_));

    size_t curr_freq = 0;
    size_t max_freq  = 0;

    double most_frequent_value = v_.front();
    double last_seen_value     = v_.front();

    for (const auto value : v_)
    {
        if (IsEqual(value, last_seen_value, last_seen_value * EPS_R)) ++curr_freq;
        else
        {
            if (curr_freq > max_freq)
            {
                max_freq = curr_freq;
                most_frequent_value = last_seen_value;
            }

            last_seen_value = value;
            curr_freq = 1;
        }
    }

    if (curr_freq > max_freq)
    {
        max_freq = curr_freq;
        most_frequent_value = last_seen_value;
    }

    return std::make_pair(max_freq, most_frequent_value);
}


void make_tree(T_TREE & tree, const T_VEC_BIT & global_bit, const T_VEC_32 & scale_, const T_MAT_D & train_this, const T_VEC_D & train_last_el_this, const T_VEC_SIZE & histogram, const T_VEC_SIZE & train_last_id, const bool uses_train_id_lines = false, const int32_t p_id = 0)
{
    T_VEC_BIT c(train_this[0].size() - 1, false);
    T_VEC_BIT r(global_bit);

    std::unordered_map<uint64_t, size_t> hash_col;

    // T_VEC_D w_col(c.size(), 1.0);

    for (size_t i = 0; i < c.size(); i++)
        c[i] = (scale_[i] == 2);

    // print_vector(scale_);

    std::queue<T_VEC_BIT> queue_c, queue_r;
    std::queue<uint32_t> queue_tree;

    queue_tree.push(0);

    uint32_t temp_tree;

    // T_DOUBLE_SET set_results = {};

    double opt = MAGIC_1;

    for(;;)
    {
        if (!queue_tree.empty())
        {
            temp_tree = queue_tree.front();
            queue_tree.pop();
        }
        else return;

        if (!queue_c.empty())
        {
            c = queue_c.front();
            queue_c.pop();
        }

        if (!queue_r.empty())
        {
            r = queue_r.front();
            queue_r.pop();
        }

        auto leva = tree[temp_tree].level;

        if (leva + 1 > MAX_LEVEL)
        {
            endlich(tree[temp_tree], r, train_last_id, uses_train_id_lines);
            continue;
        }

        T_VEC_32 v;
        v.reserve(c.size());

        size_t i = 0;
        for (i = 0; i < c.size(); i++)
        {
            if (!c[i]) v.push_back(i);
        }

        // print_vector(c);

        if (v.size() == 0)
        {
            endlich(tree[temp_tree], r, train_last_id, uses_train_id_lines);
            continue;
        }

        auto los = truly_rand(1, 3, p_id);

        if (HISTOGRAM > 0) los = HISTOGRAM;

        size_t col_;

        switch(los) 
        {
            case 1: col_ = rand_element(v, p_id); break;
            case 2: 
            {      
                T_VEC_SIZE h;

                for (i = 0; i < c.size(); i++)
                {
                    if (!c[i] && (histogram[i] > 0)) 
                    for (size_t j = 0; j < histogram[i]; j++) h.push_back(i);
                } 

                col_ = rand_element(h, p_id);

                break;
            }
            case 3: 
            {      
                T_VEC_SIZE h;

                for (i = 0; i < c.size(); i++)
                {
                    if (!c[i] && (histogram[i] > 0)) 
                    for (size_t j = 0; j < std::log2(histogram[i]); j++) h.push_back(i);
                } 

                col_ = rand_element(h, p_id);

                break;  
            }
        }

        /*

        auto father_col = tree[temp_tree].col;

        if (father_col != -1)
        {
            auto count_col_ = hash_col[pack2(father_col, col_)];

            if (count_col_ > 0)
                for (auto c_ : knuth_shuffle(v, p_id))
                {
                    auto key_ = pack2(father_col, c_);
                    if (hash_col[key_] == 0)
                    {
                        col_ = c_;
                        break;
                    }
                }

            hash_col[pack2(father_col, col_)]++;
        }

        */

        auto scale_col = scale_[col_];

        if (scale_col == 2)
        {
            endlich(tree[temp_tree], r, train_last_id, uses_train_id_lines);
            continue;
        }

        T_VEC_D bb;
        bb.reserve(r.size());

        T_VEC_D last_full;
        last_full.reserve(r.size());

        T_VEC_32 cases;
        cases.reserve(r.size());

        double max_in_col = LOWEST_DOUBLE;
        double min_in_col = MAX_DOUBLE;

        i = 0;
        while (i < r.size())
        {
           if (r[i])
           {
               auto el = train_this[i][col_];
               if (!IsEqual(el, MISSING))
               {
                   bb.push_back(el);
                   last_full.push_back(train_last_el_this[i]);
                   cases.push_back(i);

                   if (!DISCRETIO)
                   {
                       max_in_col = std::max(max_in_col, el);
                       min_in_col = std::min(min_in_col, el);
                   }
               }
           }
           i++;
        }

        if (last_full.size() < COUNT_FROM + 1)
        {
            endlich(tree[temp_tree], r, train_last_id, uses_train_id_lines);
            continue;
        }

        std::sort( bb.begin(), bb.end() );
        bb.erase( std::unique (bb.begin(), bb.end(), [](double x1, double x2) { return IsEqual(x1, x2); } ), bb.end() );

        // std::sort( last_full.begin(), last_full.end() );
        // last_full.erase( std::unique (last_full.begin(), last_full.end(), [](double x1, double x2) { return IsEqual(x1, x2); } ), last_full.end() );

        int32_t dog_max_try_thr = bb.size() / 2;

        opt = tree[temp_tree].opt;

        if (IsEqual(opt, MAGIC_1))
        {
            opt = rand_element(last_full, p_id);
            tree[temp_tree].opt = opt;
        }

        double best_threshold;

        std::pair<double, double> r2_r, r2_l;

        size_t sz_r, sz_l;

        T_VEC_BIT new_r_left(r.size(), false);
        T_VEC_BIT new_r_right(r.size(), false);

        T_VEC_32 tmp_l, tmp_r;

        for(;;)
        {
            int32_t criterion_ = 0;

            if (dog_max_try_thr-- < 0) break;

            if (dog_max_try_thr == 0) criterion_ = 2; // truly_rand(1, 2, p_id);

            if ( !DISCRETIO && (scale_col == 0) )
                best_threshold  = truly_rand_real(min_in_col, max_in_col, p_id);
            else
                best_threshold = rand_element(bb, p_id);

            if (criterion_ > 0)
            {
                double bestGain = 0.0;

                switch (criterion_)
                {
                    case 1:

                        for (size_t it_b = 0; it_b < bb.size(); it_b++)
                        {
                            auto rr = bb[it_b];

                            T_VEC_D e_l, e_r;

                            for (auto i_ : cases)
                            {
                                auto local_ = train_this[i_][col_];
                                auto el = train_last_el_this[i_];

                                bool b = (scale_col == 1) ? (IsEqual(local_, rr)) : (local_ >= rr);

                                if (b)
                                    e_r.push_back(el);
                                else
                                    e_l.push_back(el);
                            }

                            double LH = res_r(e_l);
                            if (IsEqual(LH, MAGIC_1)) continue;

                            double RH = res_r(e_r);
                            if (IsEqual(RH, MAGIC_1)) continue;

                            if (IsEqual(LH, RH)) continue;

                            double informationGain = std::fabs(LH - RH);

                            if (  (informationGain > bestGain) || (it_b == 0)  )
                            {
                                bestGain = informationGain;
                                best_threshold = rr;
                            }
                        }

                        break;

                    case 2:

                        auto p = counter_(last_full, opt, opt * EPS_R);

                        if (IsEqual(p, 0.0))
                        {
                            opt = rand_element(last_full, p_id);
                            tree[temp_tree].opt = opt;
                            p = counter_(last_full, opt, opt * EPS_R);
                        }

                        auto N = last_full.size();
                        p = (1.0 + p) / (N + 2.0);
                        double q = (1.0 + N - p) / (N + 2.0);

                        auto H = -p * std::log(p) - q * std::log(q);

                        for (size_t it_b = 0; it_b < bb.size(); it_b++)
                        {
                            auto rr = bb[it_b];

                            T_VEC_D e_l, e_r;

                            for (auto i_ : cases)
                            {
                                auto local_ = train_this[i_][col_];
                                auto el = train_last_el_this[i_];

                                bool b = (scale_col == 1) ? (IsEqual(local_, rr)) : (local_ >= rr);

                                if (b)
                                    e_r.push_back(el);
                                else
                                    e_l.push_back(el);
                            }

                            double l_yes = counter_(e_l, opt, opt * EPS_R);
                            double r_yes = counter_(e_r, opt, opt * EPS_R);

                            // if (IsEqual(l_yes, r_yes)) continue;

                            double l_no = e_l.size() - l_yes;
                            double r_no = e_r.size() - r_yes;

                            l_yes += 1.0;
                            l_no += 1.0;

                            double t1 = l_yes + l_no;

                            r_yes += 1.0;
                            r_no += 1.0;

                            double t2 = r_yes + r_no;

                            l_yes /= t1;
                            l_no /= t1;

                            r_yes /= t2;
                            r_no /= t2;

                            double LH = -l_yes * std::log(l_yes) - l_no * std::log(l_no);
                            double RH = -r_yes * std::log(r_yes) - r_no * std::log(r_no);

                            double informationGain = H - LH - RH;

                            if (  (informationGain > bestGain) || (it_b == 0)  )
                            {
                                bestGain = informationGain;
                                best_threshold = rr;
                            }
                        }

                        break;
                }
            }

            T_VEC_D e_l, e_r;

            e_l.reserve(cases.size());
            e_r.reserve(cases.size());

            tmp_l.clear();
            tmp_r.clear();

            tmp_l.reserve(cases.size());
            tmp_r.reserve(cases.size());

            std::fill(new_r_left.begin(),  new_r_left.end(),  false);
            std::fill(new_r_right.begin(), new_r_right.end(), false);

            for (auto i_ : cases)
            {
                auto local_ = train_this[i_][col_];
                auto el = train_last_el_this[i_];

                bool b = (scale_col == 1) ? (IsEqual(local_, best_threshold)) : (local_ >= best_threshold);

                if (b)
                {
                    new_r_right[i_] = true;
                    tmp_r.push_back(i_);
                    e_r.push_back(el);
                }
                else
                {
                    new_r_left[i_] = true;
                    tmp_l.push_back(i_);
                    e_l.push_back(el);
                }
            }

            sz_l = tmp_l.size();
            sz_r = tmp_r.size();

            r2_r = average_variance(e_r);
            auto fq_r = IsEqual(tree[temp_tree].res, r2_r.first, (tree[temp_tree].res + r2_r.first) * 0.5 * EPS_R);

            r2_l = average_variance(e_l);
            auto fq_l = IsEqual(tree[temp_tree].res, r2_l.first, (tree[temp_tree].res + r2_l.first) * 0.5 * EPS_R);


            if (fq_r || fq_l) continue;

            /*
            if (hash_col[
                         //xor64_bits(cut64_bits(hash64(r2_r.first)), sz_r) ^
                         //xor64_bits(cut64_bits(hash64(r2_l.first)), sz_l) ^
                         //xor64_bits(cut64_bits(hash64(best_threshold)), col_) ^
                         //xor64_bits(cut64_bits(hash64(tree[temp_tree].res)), tree[temp_tree].col)

                         //cut64_bits(hash64(r2_r.first)) ^
                         //cut64_bits(hash64(r2_l.first)) ^

                         //cut64_bits(hash64(tree[temp_tree].res))
                         //xor64_bits(cut64_bits(hash64(best_threshold)), col_)
                        ]++ > 0) continue;
            */

            // if (!is_compress(e_l, e_r)) continue;

            break;
        }

        if (dog_max_try_thr < 0)
        {
            endlich(tree[temp_tree], r, train_last_id, uses_train_id_lines);
            continue;
        }

        T_VEC_BIT new_c(c);

        if (!OBLIVIOUS) new_c[col_] = true; // no repeat cols

        size_t count_new_c = std::count(new_c.begin(), new_c.end(), true);
        bool end_new_c = (count_new_c == new_c.size()) || (count_new_c > MAX_WIDTH); //truly_rand(1, MAX_WIDTH, p_id) );

        bool flag_r = true;

        if (sz_r >= COUNT_FROM)
        {
            if (sz_r <= COUNT_TO)
            {
                if (r2_r.second < VARIANCE_RESULT)
                {
                    BinaryTree t_;
                    t_.parent = temp_tree;
                    t_.col = col_;
                    t_.threshold = best_threshold;
                    t_.opt = opt;
                    t_.flag = 2;
                    t_.res = r2_r.first;
                    t_.size_res = sz_r;
                    t_.level = tree[temp_tree].level + 1;

                    if (uses_train_id_lines)
                    {
                        double ww = 1.0;
                        for (auto i : tmp_r)
                        {
                            std::lock_guard<std::mutex> lock(barrier_cases_id);
                            ww += (double)++global_uses_train_lines[train_last_id[i]];
                        }
                        ww = std::log(ww);
                        t_.early = 1.0 / ( ww * ww );
                    }

                    tree.push_back(t_);
                    tree[temp_tree].right = tree.size() - 1;
                    flag_r = false;
                }
            }
            else
            {
                if (!end_new_c)
                {
                    BinaryTree t_;
                    t_.parent = temp_tree;
                    t_.col = col_;
                    t_.threshold = best_threshold;
                    t_.opt = opt;
                    t_.flag = 1;
                    t_.res = r2_r.first;
                    t_.size_res = sz_r;
                    t_.level = tree[temp_tree].level + 1;
                    tree.push_back(t_);
                    tree[temp_tree].right = tree.size() - 1;

                    queue_c.push(new_c);
                    queue_r.push(new_r_right);
                    queue_tree.push(tree[temp_tree].right);
                    flag_r = false;
                }
            }
        }

        bool flag_l = true;

        if (sz_l >= COUNT_FROM)
        {
            if (sz_l <= COUNT_TO)
            {
                if (r2_l.second < VARIANCE_RESULT)
                {
                    BinaryTree t_;
                    t_.parent = temp_tree;
                    t_.col = col_;
                    t_.threshold = best_threshold;
                    t_.opt = opt;
                    t_.flag = 2;
                    t_.res = r2_l.first;
                    t_.size_res = sz_l;
                    t_.level = tree[temp_tree].level + 1;

                    if (uses_train_id_lines)
                    {
                        double ww = 1.0;
                        for (auto i : tmp_l)
                        {
                            std::lock_guard<std::mutex> lock(barrier_cases_id);
                            ww += (double)++global_uses_train_lines[train_last_id[i]];
                        }
                        ww = std::log(ww);
                        t_.early = 1.0 / ( ww * ww );
                    }

                    tree.push_back(t_);
                    tree[temp_tree].left = tree.size() - 1;
                    flag_l = false;
                 }
            }
            else
            {
                if (!end_new_c)
                {
                    BinaryTree t_;
                    t_.parent = temp_tree;
                    t_.col = col_;
                    t_.threshold = best_threshold;
                    t_.opt = opt;
                    t_.flag = 1;
                    t_.res = r2_l.first;
                    t_.size_res = sz_l;
                    t_.level = tree[temp_tree].level + 1;
                    tree.push_back(t_);
                    tree[temp_tree].left = tree.size() - 1;

                    queue_c.push(new_c);
                    queue_r.push(new_r_left);
                    queue_tree.push(tree[temp_tree].left);
                    flag_l = false;
                }
            }
        }

        if (flag_l && flag_r)
            endlich(tree[temp_tree], r, train_last_id, uses_train_id_lines);
    }
}


resTree make_predict(const T_TREE & tree, const size_t t, const size_t row, const int32_t dim = 0, const int32_t lr = 0, const bool is_missing = false)
{
    resTree rrr;

    if (tree[t].flag == 3) return rrr;

    if (lr == 0)
    {
        resTree rrr1 = make_predict(tree, tree[t].left, row, dim, -1);
        resTree rrr2;

        if (!rrr1.flag || is_missing)
            rrr2 = make_predict(tree, tree[t].right, row, dim, 1);

        /*
        std::stringstream o;

        o << std::boolalpha << rrr1.flag << " --- " << std::boolalpha << rrr2.flag << std::endl;

        print_log(o.str());
        */

        if (rrr1.flag && rrr2.flag)
        {
            resTree rrr3;
            rrr3.res = (rrr1.res + rrr2.res) * 0.5;

            if (IsEqual(rrr1.res, rrr2.res, rrr3.res * EPS_R))
            {
               rrr3.flag = true;

               rrr3.id1 = rrr1.id1;
               rrr3.id2 = rrr2.id1;

               rrr3.w   = (rrr1.w + rrr2.w) * 0.5;
            }
            return rrr3;
        }

        if (rrr1.flag) return rrr1;
        if (rrr2.flag) return rrr2;
        return rrr;

    }
    else
    {
        auto select_col = tree[t].col;

        if (select_col == -1) return rrr; // print_value(tree[t].level, "is");

        double el = MISSING;
        switch (dim)
        {
            case 0: el = test[row][select_col]; break;
            case 1: el = control[row][select_col]; break;
            case 2: el = train[row][select_col]; break;
            case 3: el = holdout[row][select_col]; break;
        }

        auto fail_ = IsEqual(el, MISSING);

        // if (fail_) return rrr;

        auto scale_col = scale[select_col];
        bool b = (scale_col == 1) ? (IsEqual(el, tree[t].threshold)) : (el >= tree[t].threshold);

        if  (
              ((lr == -1) && !b) || ((lr == 1) && b) || fail_
            )
        {

            if (tree[t].flag == 2)
            {
                auto col_ = tree[t].col;

                switch (dim)
                {
                    case 0:
                    {
                        std::lock_guard<std::mutex> lock(barrier_test);
                        hash_test_col[col_]++;
                        break;
                    }
                    case 3:
                    {
                        std::lock_guard<std::mutex> lock(barrier_holdout);
                        hash_holdout_col[col_]++;
                        break;
                    }
                }
            }

            switch (tree[t].flag)
            {
                case 1: return make_predict(tree, t, row, dim, 0, fail_);
                case 2:
                {
                    auto r_ = tree[t].res;

                    rrr.flag = !IsEqual(r_, MAGIC_1);
                    rrr.res = r_;
                    rrr.id1 = rrr.id2 = tree[t].id;
                    if (W == 1) rrr.w = correction(tree[t].size_res) * std::exp(-(double) tree[t].level * tree[t].early * 0.5); // 0.22464

                    return rrr;
                }
            }
        }
    }

    return rrr;
}

void grow(const size_t a, const size_t b, T_FOREST & Forest, const T_MAT_D & train_this, const T_VEC_D & train_last_el_this, const T_VEC_SIZE & train_last_id, const bool uses_train_id_lines = false, const int32_t p_id = 0)
{
    auto nn_r = train_this.size();
    auto nn_c = train_this[0].size();

    T_TREE stree_clear; BinaryTree t_; stree_clear.push_back(t_);

    for (size_t j = a; j <= b; j++)
       Forest[j] = stree_clear;

    size_t false_count = 0;

    T_VEC_BIT global_bit(train_this.size(), true);

    for (size_t j = 0; j < train_last_el_this.size(); j++)
    {
        bool f = IsEqual(train_last_el_this[j], MAGIC_1) || IsEqual(train_last_el_this[j], MISSING);
        global_bit[j] = !f;
    }

    auto scale_(scale_epoch);

    if (FILTER_GROW.size() > 0)
    {

        /*
        for (size_t j = 0; j < 10; j++)
        {
            auto kv = rand_element(FILTER_GROW, p_id);
            std::string s =  std::to_string(j) + ": " +  std::to_string(kv.first) + " : " +  std::to_string(kv.second);
            print_log(s);
        }

        exit(0);
        */

        auto kv = rand_element(FILTER_GROW, p_id);

        scale_[kv.first] = 2;

        for (size_t row = 0; row < nn_r; row++)
            global_bit[row] = IsEqual(kv.second, train_this[row][kv.first]);
    }

    T_VEC_SIZE histogram;

    if (HISTOGRAM != 1)
    {
        histogram.resize(nn_c, 0); 

        for (size_t col = 0; col < nn_c; col++)
        {

            T_VEC_SIZE v;
            v.reserve(nn_c); 

            for (size_t row = 0; row < nn_r; row++)
            {
                if (!global_bit[row]) continue;
                v.push_back(train_this[row][col]);
            }

            std::sort( v.begin(), v.end() );
            v.erase( std::unique (v.begin(), v.end(), [](double x1, double x2) { return IsEqual(x1, x2); } ), v.end() );

            histogram[col] = v.size();
        }
    } 
 
    size_t j = a;
    for (;;)
    {
        false_count = std::count(global_bit.begin(), global_bit.end(), false);
        if ( false_count == nn_r ) break;

        T_TREE tree;
        BinaryTree t_; t_.flag = 0; tree.push_back(t_);

        make_tree(tree, global_bit, scale_, train_this, train_last_el_this, histogram, train_last_id, uses_train_id_lines, p_id);

        do {} while (delete_trash(tree));

        if (tree.size() < 1) continue;
        if (tree[0].flag == 3) continue;

        Forest[j++] = tree;

        if (j > b) break;

        if (EXCLUDE_SUCCESS)
        for (size_t row = 0; row < nn_r; row++)
        {
            if (!global_bit[row]) continue;
            auto r = make_predict(tree, 0, row, 2);
            if (r.flag)
            {
                if (IsEqual(r.res, train_last_el_this[row]), train_last_el_this[row] * EPS_R) global_bit[row] = false;
            }
        }
    }
}

// http://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const std::vector<T> & vec, Compare & compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(const std::vector<T>& vec, const std::vector<std::size_t> & p)
{
    std::vector<T> sorted_vec(p.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

T_MAT_D transpose(const T_MAT_D & data)
{
    T_MAT_D result(data[0].size(), T_VEC_D (data.size()));

    for (size_t i = 0; i < data[0].size(); i++)
        for (size_t j = 0; j < data.size(); j++)
            result[i][j] = data[j][i];

    return result;
}

bool cmp_double(const double & a, const double & b) { return (a < b); }

bool cmp_double2(const double & a, const double & b) { return (a > b); }

void predict(const size_t a, const size_t b, const T_FOREST & Forest, T_MAT_D & res_this, T_MAT_D & w_this, const int32_t dim = 0)
{
    const bool exist_mat_id = MAT_ID.size() > 0;
    const auto sz = Forest.size();

    for (size_t i = a; i <= b; i++)
    {
        T_VEC_SIZE vs;

        if (exist_mat_id)
            vs.reserve(sz * 2);

        T_VEC_D p_r(sz, MAGIC_1);
        T_VEC_D p_w(sz, MAGIC_1);

        for (size_t j = 0; j < sz; j++)
        {
            resTree r = make_predict(Forest[j], 0, i, dim);

            if (r.flag)
            {
                p_r[j] = r.res;
                p_w[j] = r.w;

                if (exist_mat_id)
                {
                    const auto id1 = r.id1;
                    const auto id2 = r.id2;

                    vs.push_back(id1);
                    if (id1 != id2)
                        vs.push_back(id2);
                }
            }
        }

        auto vs_size = vs.size();
        if ( exist_mat_id && (vs_size > 0) )
        {
            std::lock_guard<std::mutex> lock(barrier);
            MAT_ID[i].reserve(MAT_ID[i].size() + vs_size);
            MAT_ID[i].insert(MAT_ID[i].end(), vs.begin(), vs.end());
        }

        res_this[i] = p_r;
        w_this[i]   = p_w;
    }
}


T_PRED predict_(const T_FOREST & F, const int32_t dim = 0)
{
    size_t size_v;

    switch (dim)
    {
        case 0: size_v = test.size(); break;
        case 1: size_v = control.size(); break;
        case 2: size_v = train.size(); break;
        case 3: size_v = holdout.size(); break;
    }

    T_MAT_D res_(size_v);
    T_MAT_D   w_(size_v);

    T_VEC_D r_(size_v, MAGIC_1);

    size_t n_threads_ = n_threads;

    if (n_threads > size_v)
        n_threads_ = size_v;

    T_VEC_32 limits = bounds(n_threads_, size_v);

    std::vector<std::thread> th;

    for (size_t id_ = 0; id_ < n_threads_; id_++)
        th.push_back(std::thread(predict, limits[id_], limits[id_ + 1] - 1, F, std::ref(res_), std::ref(w_), dim));

    for (auto &t : th) t.join();

    auto e_len = res_[0].size();


    T_VEC_32 ia, ib;

    for (size_t i = 0; i < e_len; i++)
        for (size_t j = i + 1; j < e_len; j++)
        {
             ia.push_back(i);
             ib.push_back(j);
        }

    const size_t ab_len = ia.size();

    T_VEC_D permit_p(ab_len);

    {

    size_t n_threads_ = n_threads;

    if (n_threads > ab_len)
        n_threads_ = ab_len;

    T_VEC_32 limits = bounds(n_threads_, ab_len);

    std::vector<std::thread> th;
    for (size_t id_ = 0; id_ < n_threads_; id_++)
        th.push_back(std::thread(make_permit, limits[id_], limits[id_ + 1] - 1, std::ref(ia), std::ref(ib), std::ref(res_), std::ref(permit_p)));

    for (auto &t : th) t.join();

    }

    T_VEC_D dd(e_len, 0.0);

    for (size_t i = 0; i < ab_len; i++)
    {
        auto t = std::fabs(permit_p[i]);

        dd[ia[i]] += t;
        dd[ib[i]] += t;
    }

    auto o = sort_permutation(dd, cmp_double2);

    res_ = transpose(res_);

    res_ = apply_permutation(res_, o);

    auto r_s = std::round(res_.size() * CUT_PREDICT);

    res_.resize(r_s);

    w_ = transpose(w_);

    w_ = apply_permutation(w_, o);

    w_.resize(r_s);

    T_VEC_D    ww(size_v, 0.0);
    T_VEC_SIZE pp(size_v, 0);

    for (size_t i = 0; i < r_s; i++)
        for (size_t j = 0; j < size_v; j++)
        {
            auto e = res_[i][j];
            auto w =   w_[i][j];

            if (W == 0) w = 1.0;

            if ( !IsEqual(e, MAGIC_1) )
            {
                if ( IsEqual(r_[j], MAGIC_1) ) r_[j] = e * w; else r_[j] += e * w;
                ww[j] += w;
                pp[j]++;
            }
        }

    double w_res = 0.0;
    size_t n_ww = 0;

    for (size_t j = 0; j < size_v; j++)
    {
        auto w = ww[j];
        auto p = pp[j];

        if ( (!IsEqual(w, 0.0)) and (p > 0) )
        {
            r_[j] /= w;
            double a = w / p;
            w_res += a * a;
            n_ww++;
        }
    }

    if (n_ww > 0) w_res /= (double) n_ww;

    return std::make_pair(r_, w_res);
}


double find_PAIR(const int32_t & key_, const T_VEC_PAIR & vpair)
{
    double r = 0.0;
    auto ind_ = std::find_if( vpair.begin(), vpair.end(), [& key_](const T_PAIR & el) { return el.first == key_; } );
    if ( ind_ != vpair.end() ) r = ind_->second;
    return r;
}

double get_min(const T_VEC_D & a, const T_VEC_D & b)
{
    double m_ = MAX_DOUBLE;
    double sign_ = 1.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        double dab = a[i] - b[i];
        if (!IsEqual(dab, 0.0))
        {
            m_  = std::min(m_, std::fabs(dab));
            if (dab < 0.0) sign_ = -1.0;
        }
    }
    return sign_ * m_;
}

// https://www.kaggle.com/c/SemiSupervisedFeatureLearning/forums/t/919/auc-implementation

double roc_auc_score(const T_VEC_D & actual_, const T_VEC_D & pred_)
{
   // From 'AUC Calculation Check' post in IJCNN Social Network Challenge forum
   // Credit: B Yang - original C++ code
    auto p = sort_permutation(pred_, cmp_double); // [](const double & a, const double & b){ return (a < b); });

    auto actual = apply_permutation(actual_, p);
    auto pred = apply_permutation(pred_, p);

    size_t n = actual.size();
    double ones = std::accumulate(std::begin(actual), std::end(actual), 0.0);
    if (IsEqual(ones, 0.0) || IsEqual(ones, (double)n)) return 1.0;

    double tp0, tn;
    double truePos = tp0 = ones;
    double accum = tn = 0.0;
    double threshold = pred[0];

    for (size_t i = 0; i < n; i++)
    {
        if (!IsEqual(pred[i], threshold))
        { // threshold changes
            threshold = pred[i];
            accum += tn * (truePos + tp0); //2* the area of  trapezoid
            tp0 = truePos;
            tn = 0.0;
        }
        tn += 1.0 - actual[i]; // x-distance between adjacent points
        truePos -= actual[i];
    }

    accum += tn * (truePos + tp0); // 2 * the area of trapezoid
    return (double)accum / (2.0 * ones * ((double)n - ones));
}

double log_loss(const T_VEC_D & actual_, const T_VEC_D & pred_)
{
    double r = 0.0;
    auto n = actual_.size();
    for (size_t i = 0; i < n; i++)
    {
        auto a = actual_[i];
        auto p = std::min(1.0 - EPS, std::max(EPS, pred_[i]));
        r += a * std::log(p) + (1.0 - a) * std::log(1.0 - p);
    }
    return -r / (double) n;
}

double square_loss(const T_VEC_D & actual_, const T_VEC_D & pred_)
{
    double r = 0.0;
    auto n = actual_.size();
    for (size_t i = 0; i < n; i++)
    {
        auto t = actual_[i] - pred_[i];
        r += t * t;
    }
    return r / (double) n;
}


double rmspe_loss(const T_VEC_D & actual_, const T_VEC_D & pred_)
{
    double r = 0.0;
    auto n = actual_.size();

    for (size_t i = 0; i < n; i++)
    {
        auto t = (actual_[i] - pred_[i]) / (actual_[i] + EPS);
        r += t * t;
    }

    return std::sqrt(r / (double) n);
}

double prepare_loss(const T_VEC_D & actual_, const T_VEC_D & pred_)
{
    T_VEC_D actual, pred;

    auto n_a = actual_.size();
    auto n_p = pred_.size();

    if (n_a != n_p) return MAGIC_1;

    for (size_t i = 0; i < n_a; i++)
    {
        if (!IsEqual(pred_[i], MAGIC_1))
        {
            actual.push_back(actual_[i]);
            pred.push_back(pred_[i]);
        }
    }

    n_a = actual.size();
    n_p = pred.size();

    if ( (n_a == 0) || (n_p == 0) ) return MAGIC_1;

    double LOSS_ADD = 0.0;
    switch (LOSS)
    {
        case 0: LOSS_ADD = square_loss(actual, pred); break;
        case 1: LOSS_ADD = log_loss(actual, pred); break;
        case 2: LOSS_ADD = rmspe_loss(actual, pred); break;
        case 3: LOSS_ADD = roc_auc_score(actual, pred); break;
    }

    return LOSS_ADD;
}


void train_and_test_and_holdout_parody()
{
    const size_t size_train   = 12000;
    const size_t size_test    = 2000;
    const size_t size_holdout = 2000;
    const size_t size_cols    = 10;
    const size_t size_val     = 50;

    std::ofstream outtrain(TRAIN_FILE);

    for(size_t row = 0; row < size_train; row++)
    {
        std::ostringstream s;
        s.clear();

        int32_t sum, sum2;
        sum = sum2 = 0;

        for (size_t col = 0; col < size_cols; col++)
        {
            int32_t t, t2;
            t = t2 = truly_rand(0, size_val);
            if (truly_rand01() < 0.1)
            {
                switch (NOISE_MISS)
                {
                    case 1 : t2 = truly_rand(0, size_val); break;
                    case 2 : t2 = MISSING; break;
                    case 3 : t2 = (truly_rand01() < 0.5) ? truly_rand(0, size_val) : MISSING; break;
                }
            }

            if ( (LEAK_TEST == 1) && (col == 9) )
            {

            }
            else
            {
                s << t2 << ",";
            }

            sum += t;
            sum2 += truly_rand(0, size_val);
        }

        if (BINARY_TEST == 1)
        {
            sum  = ( (size_val * size_cols / 2) > sum  ) ? 1.0 : 0.0;
            sum2 = ( (size_val * size_cols / 2) > sum2 ) ? 1.0 : 0.0;
        }

        if (LEAK_TEST == 1)
        {
            s << sum << ",";
        }

        if ( ( (NOISE_MISS == 1) || (NOISE_MISS == 3) ) && (truly_rand01() < 0.1) )
        {
            s << sum2 << std::endl;
        }
        else
        {
            s << sum << std::endl;
        }

        outtrain << s.str();
    }

    outtrain.close();

    std::ofstream outtest(TEST_FILE);

    T_VEC_D a, b, c, i;

    a.resize(size_test, 0.0);
    b.resize(size_test, 0.0);
    c.resize(size_test, 0.0);
    i.resize(size_test, 0.0);

    for(size_t row = 0; row < size_test; row++)
    {
        std::ostringstream s;
        s.clear();

        int32_t sum1, sum2, sum3;
        sum1 = sum2 = sum3 = 0;

        for(size_t col = 0; col < size_cols; col++)
        {
            int32_t t, t2;
            t = t2 = truly_rand(0, size_val);
            if (truly_rand01() < 0.1)
            {
                switch (NOISE_MISS)
                {
                    case 1 : t2 = truly_rand(0, size_val); break;
                    case 2 : t2 = MISSING; break;
                    case 3 : t2 = (truly_rand01() < 0.5) ? truly_rand(0, size_val) : MISSING; break;
                }
            }

            if ( (LEAK_TEST == 1) && (col == 9) )
            {

            }
            else
            {
                s << t2 << ",";
            }

            sum1 += t;
            sum2 += truly_rand(0, size_val);

            if (!IsEqual(t2, MISSING))
                sum3 += t2;
            else
                sum3 += truly_rand(0, size_val);
       }

        a[row] = sum2;
        b[row] = sum1;
        c[row] = sum3;

        if (BINARY_TEST == 1)
        {
            a[row] = ( (size_val * size_cols / 2) > a[row] ) ? 1.0 : 0.0;
            b[row] = ( (size_val * size_cols / 2) > b[row] ) ? 1.0 : 0.0;
            c[row] = ( (size_val * size_cols / 2) > c[row] ) ? 1.0 : 0.0;
        }

        if (LEAK_TEST == 1)
        {
            s << b[row] << ",";
        }

        if ( ( (NOISE_MISS == 1) || (NOISE_MISS == 3) ) && (truly_rand01() < 0.1) )
        {
            s << a[row] << std::endl;
            i[row] = a[row];
        }
        else
        {
            s << b[row] << std::endl;
            i[row] = b[row];
        }

        outtest << s.str();
    }

    outtest.close();

    auto loss_1 = prepare_loss(c, b);
    auto loss_2 = prepare_loss(i, b);
    auto loss_3 = prepare_loss(a, b);

    std::cout << std::endl
    << "ideal loss (parody): "
    << make_colored("", "31", 1) << loss_1 << make_colored("", "31", 2) << " - "
    << make_colored("", "31", 1) << loss_2 << make_colored("", "31", 2) << " - "
    << make_colored("", "31", 1) << loss_3 << make_colored("", "31", 2)
    << std::endl;

    std::ofstream outval(HOLDOUT_FILE);

    for(size_t row = 0; row < size_holdout; row++)
    {
        std::ostringstream s;
        s.clear();

        int32_t sum, sum2;
        sum = sum2 = 0;

        for(size_t col = 0; col < size_cols; col++)
        {
            int32_t t, t2;
            t = t2 = truly_rand(0, size_val);
            if (truly_rand01() < 0.1)
            {
                switch (NOISE_MISS)
                {
                    case 1 : t2 = truly_rand(0, size_val); break;
                    case 2 : t2 = MISSING; break;
                    case 3 : t2 = (truly_rand01() < 0.5) ? truly_rand(0, size_val) : MISSING; break;
                }
            }

            if ( (LEAK_TEST == 1) && (col == 9) )
            {

            }
            else
            {
                s << t2 << ",";
            }

            sum += t;
            sum2 += truly_rand(0, size_val);
        }

        if (BINARY_TEST == 1)
        {
            sum  = ( (size_val * size_cols / 2) > sum  ) ? 1.0 : 0.0;
            sum2 = ( (size_val * size_cols / 2) > sum2 ) ? 1.0 : 0.0;
        }

        if (LEAK_TEST == 1)
        {
            s << sum << ",";
        }

        if ( ( (NOISE_MISS == 1) || (NOISE_MISS == 3) ) && (truly_rand01() < 0.1) )
        {
            s << sum2 << std::endl;
        }
        else
        {
            s << sum << std::endl;
        }

        outval << s.str();
    }

    outval.close();
}



int32_t main(int argc, char* argv[])
{

  /*

   DEBUG_MODE = true;

   // cout << "1" << endl;

   scale = split32(read_tag("scale", "", "\"0,\"*100"), ',');

   load_test("t.csv");

    size_t i = 0;
    for (auto & e : test)
    {
        size_t j = 0;
        for (auto & x : e)
        {
             temp[i][j]
             j++;
        }
        i++;
    }

    exit(0);

    */

    /*
    const std::string st = "12:23;4:6-10;2:2-3";

    std::cout << st << std::endl;

    auto t = splitPAIR(st);

    for (auto & xx : t)
        std::cout << xx.first << " : " << xx.second << std::endl;

    exit(0);
    */

    if (argc < 2)
    {
        std::cout << std::endl
        << make_colored(about)
        << std::endl << std::endl <<

        "    How use: ./brf.o myconfig.xml\n\n"
        "    Example xml:\n\n"

        "<test_estimate>true</test_estimate>\n"
        "<missing>77877.0</missing>\n\n"

        "<eps_r>0.0001</eps_r>\n\n"

        "<max_width>2000</max_width>\n"
        "<max_level>6</max_level>\n"
        "<count_from>10</count_from>\n"
        "<count_to>100</count_to>\n"
        "<n_forest>100</n_forest>\n"
        "<n_forest_err>100</n_forest_err>\n"
        "<n_forest_err2>100</n_forest_err2>\n"
        "<dog>50</dog>\n"
        "<epoch>2</epoch>\n"
        "<iter_err>10</iter_err>\n\n"

        "<boost>0</boost>\n"
        "<exclude_success>false</exclude_success>\n"
        "<no_control>false</no_control>\n\n"

        "<load_big>false</load_big>\n\n"

        "<mean_model>0</mean_model>\n"
        "     <-- 0 - average,\n"
        "         1 - median,\n"
        "         2 - shoorygin,\n"
        "         3 - geometric mean\n\n"

        "<hide_cols_epoch>0</hide_cols_epoch>\n"
        "<filter_grow>0:2;2:4-10</filter_grow>\n"
        "     <-- variable:value;variable:value_1-value_k\n\n"

        "<train_file>train_parody.csv</train_file>\n"
        "<test_file>test_parody.csv</test_file>\n"
        "<holdout_file>holdout_parody.csv</holdout_file>\n"
        "<result_file>result_parody.csv</result_file>\n"
        "<forest_file>forest_parody.csv</forest_file>\n"
        "<log_file>log_parody.csv</log_file>\n"

        "<scale>\"0,\"*100</scale>\n"
        "     <-- 0 - ordinal or interval scale, 1 - nominal scale, 2 - skip\n\n"

        "<shuffle>\"0,\"*100</shuffle>\n"

        "<loss>0</loss>\n"
        "     <-- 0 - squared loss,\n"
        "         1 - logarithmic loss,\n"
        "         2 - root mean square percentage error,\n"
        "         3 - area under curve\n\n"

        "<criterion>0</criterion>\n"
        "     <-- 0 - none,\n"
        "         1 - max distance,\n"
        "         2 - binary\n\n"

        "<barrier_loss>0.0</barrier_loss>\n"
        "<barrier_corr_var>0.8</barrier_corr_var>\n"
        "<barrier_corr_res>0.8</barrier_corr_res>\n\n"

        "<barrier_w>1.0</barrier_w>\n\n"

        "<min_or_max>2<min_or_max> <-- 1 or 2\n\n"

        "<cut_predict>0.33</cut_predict>\n\n"

        "<debug_mode>false</debug_mode>\n"
        "<utilize_cores>20</utilize_cores>\n"
        "<zeitnot>60</zeitnot>\n\n"

        "<max_rows_train_and_control>10000</max_rows_train_and_control>\n"
        "<split_for_control>0.5</split_for_control>\n\n"

        "<have_elements>30</have_elements>\n\n"

        "<fixed_output>true</fixed_output>\n\n"

        "<variance_result>2.0</variance_result>\n"
        "<variance_element>0.0</variance_element>\n"
        "<alpha>0.1</alpha>\n"
        "<lambda>0.01</lambda>\n"
        "<kickdown>0.3</kickdown>\n"
        "<gamma>0.0001</gamma>\n\n"

        "<histogram>1</histogram>\n"
        "     <-- 0 - random type,\n"
        "         1 - random uniform,\n"
        "         2 - N of unique,\n"
        "         3 - log2(N of unique)\n\n"

        "<k1>200</k1>\n"
        "<w>0</w>\n"
        "     <-- 0 - none regularization,\n"
        "         1 - leaf\n"

        "<discretio>true</discretio>\n"
        "<oblivious>true</oblivious>\n\n"

        "<max_brute_force>100000</max_brute_force>\n\n"

        "<seed>0</seed>\n"
        "<serialize>false</serialize>\n"

        << std::endl;
        return 1;
    }

    bool is_help_hash = (std::string(argv[1]) == "--helphash");
    if ( is_help_hash && (argc >= 3) )
    {
        auto s = std::string(argv[2]);
        std::cout << "hash of [" << s << "] = " << MYHash(s) << std::endl;
        return 1;
    }

    bool is_selftest = (std::string(argv[1]) == "--selftest");

    std::string cnfg_strs = "";

    if (is_selftest)
    {
        NOISE_MISS = str_to_int(argv[2], 0);

        switch (NOISE_MISS)
        {
            case 0 : std::cout << "no noise, no missing"; break;
            case 1 : std::cout << "only noise"; break;
            case 2 : std::cout << "only missing"; break;
            case 3 : std::cout << "noise + missing"; break;
        }
    }
    else
    {
        cnfg_strs = load_config(argv[1]);
    }

    SERIALIZE = str_to_bool(read_tag("serialize", cnfg_strs), true);
    TEST_ESTIMATE = str_to_bool(read_tag("test_estimate", cnfg_strs), true);
    MISSING = str_to_double(read_tag("missing", cnfg_strs), 77877.0);

    EPS_R = str_to_double(read_tag("eps_r", cnfg_strs), 0.0001);

    MAX_WIDTH = str_to_int(read_tag("max_width", cnfg_strs), 2000);
    MAX_LEVEL = str_to_int(read_tag("max_level", cnfg_strs), 6);

    COUNT_FROM = str_to_int(read_tag("count_from", cnfg_strs), 10);
    COUNT_TO = str_to_int(read_tag("count_to", cnfg_strs), 100); // 5
    N_FOREST = str_to_int(read_tag("n_forest", cnfg_strs), 100);
    N_FOREST_ERR = str_to_int(read_tag("n_forest_err", cnfg_strs), 100);
    N_FOREST_ERR2 = str_to_int(read_tag("n_forest_err2", cnfg_strs), 100);

    DOG = str_to_int(read_tag("dog", cnfg_strs), 50);
    EPOCH = str_to_int(read_tag("epoch", cnfg_strs), 5); // 20

    ITER_ERR = str_to_int(read_tag("iter_err", cnfg_strs), 10); // 20

    // if (EPOCH < 2) EPOCH = 2;

    BOOST = str_to_int(read_tag("boost", cnfg_strs), 0);

    EXCLUDE_SUCCESS = str_to_bool(read_tag("exclude_success", cnfg_strs), false);

    NO_CONTROL = str_to_bool(read_tag("no_control", cnfg_strs), false);

    if (NO_CONTROL) BOOST = 0;

    LOAD_BIG = str_to_bool(read_tag("load_big", cnfg_strs), false);

    // TREE_MUTATE = str_to_int(read_tag("tree_mutate", cnfg_strs), 0);
    // NODE_MUTATE = str_to_int(read_tag("node_mutate", cnfg_strs), 0);

    MEAN_MODEL = str_to_int(read_tag("mean_model", cnfg_strs), 1);

    HIDE_COLS_EPOCH = str_to_int(read_tag("hide_cols_epoch", cnfg_strs), 0); // 33% of col.count()

    FILTER_GROW = splitPAIR(read_tag("filter_grow", cnfg_strs));

    TRAIN_FILE = read_tag("train_file", cnfg_strs, "train_parody.csv");
    TEST_FILE  = read_tag("test_file", cnfg_strs, "test_parody.csv");
    HOLDOUT_FILE  = read_tag("holdout_file", cnfg_strs, "holdout_parody.csv");
    RESULT_FILE  = read_tag("result_file", cnfg_strs, "result_parody.csv");
    FOREST_FILE  = read_tag("forest_file", cnfg_strs, "forest_parody.dot");
    LOG_FILE  = read_tag("log_file", cnfg_strs, "log_parody.csv");

    std::ofstream logfile(LOG_FILE, std::ios_base::app | std::ios_base::out);

    logfile << std::endl << "-start- " << print_now() << " ------------------------>" << std::endl << cnfg_strs << std::endl;

    // scale = split32(read_tag("scale", cnfg_strs, "0,0,0,2,0,0,0,0,0,0,0"), ',');     // "\"0,\"*100"), ',');
    scale = split32(read_tag("scale", cnfg_strs, "\"0,\"*100"), ',');
    // scale = split32(read_tag("scale", cnfg_strs, "2,2,2,2,2,2,2,2,2,0,0"), ',');     // "\"0,\"*100"), ',');
    // scale = split32(read_tag("scale", cnfg_strs, "0,0,0,0,0,0,0,0,0,2,0"), ',');     // "\"0,\"*100"), ',');

    shuffle_ = split32(read_tag("shuffle", cnfg_strs, "\"0,\"*100"), ',');
    // shuffle_ = split32(read_tag("shuffle", cnfg_strs, "0,1,0,0,0,0,0,0,0,0,0"), ',');

    LOSS = str_to_int(read_tag("loss", cnfg_strs), 0);

    // if (BINARY_TEST == 1) LOSS = 1;

    BARRIER_LOSS = str_to_double(read_tag("barrier_loss", cnfg_strs), 1.0); // 9000.0

    BARRIER_CORR_VAR = str_to_double(read_tag("barrier_corr_var", cnfg_strs), 0.8);
    BARRIER_CORR_RES = str_to_double(read_tag("barrier_corr_res", cnfg_strs), 0.8);

    BARRIER_W = str_to_double(read_tag("barrier_w", cnfg_strs), 1.0); // 9000.0

    VARIANCE_RESULT = str_to_double(read_tag("variance_result", cnfg_strs), 2.0); // 120.0
    VARIANCE_ELEMENT = str_to_double(read_tag("variance_element", cnfg_strs), 0.0);

    int32_t m = 1;
    switch (LOSS)
    {
        case 0: m = 1; break; // square_loss
        case 1: m = 1; break; // log_loss
        case 2: m = 1; break; // rmspe_loss
        case 3: m = 2; break; // roc_auc_score
    }

    MIN_OR_MAX = str_to_int(read_tag("min_or_max", cnfg_strs), m);

    CRITERION = str_to_int(read_tag("criterion", cnfg_strs), 2);

    CUT_PREDICT = str_to_double(read_tag("cut_predict", cnfg_strs), 0.33);

    DEBUG_MODE = str_to_bool(read_tag("debug_mode", cnfg_strs), false);
    UTILIZE_CORES = str_to_int(read_tag("utilize_cores", cnfg_strs), 20);
    ZEITNOT = str_to_int(read_tag("zeitnot", cnfg_strs), 60);

    MAX_ROWS_TRAIN_AND_CONTROL = str_to_int(read_tag("max_rows_train_and_control", cnfg_strs), 10000); // 5800
    SPLIT_FOR_CONTROL = str_to_double(read_tag("split_for_control", cnfg_strs), 0.5);

    if (NO_CONTROL) SPLIT_FOR_CONTROL = 0.0;

    HAVE_ELEMENTS = str_to_int(read_tag("have_elements", cnfg_strs), 0); // 30

    FIXED_OUTPUT = str_to_bool(read_tag("fixed_output", cnfg_strs));

    seed_ = str_to_int(read_tag("seed", cnfg_strs), 2020);

    ALPHA = str_to_double(read_tag("alpha", cnfg_strs), 0.02); // 0.02
    LAMBDA = str_to_double(read_tag("lambda", cnfg_strs), 0.01);
    // KICKDOWN = str_to_double(read_tag("kickdown", cnfg_strs), 0.3);

    GAMMA = str_to_double(read_tag("gamma", cnfg_strs), 0.0001);

    K1 = str_to_double(read_tag("k1", cnfg_strs), 300.0);
    W  = str_to_int(read_tag("w", cnfg_strs), 1);

    DISCRETIO = str_to_bool(read_tag("discretio", cnfg_strs), false);
    OBLIVIOUS = str_to_bool(read_tag("oblivious", cnfg_strs), true);

    MAX_BRUTE_FORCE = str_to_int(read_tag("max_brute_force", cnfg_strs), 500000); // 300000

    HISTOGRAM = str_to_int(read_tag("histogram", cnfg_strs), 1);

    n_threads = std::thread::hardware_concurrency();
    if ((UTILIZE_CORES >= 1) && (UTILIZE_CORES < n_threads))
        n_threads = UTILIZE_CORES;

    std::uniform_int_distribution<uint64_t> unif;
    std::random_device rd;
    auto rd_ = rd();
    std::mt19937_64 engine_(rd_);

    if (seed_ != 0)
        engine_.seed(seed_);
    else
        seed_ = rd_;

    std::function<uint64_t()> rnd = std::bind(unif, engine_);

    for (size_t k = 0; k < n_threads + 1; k++)
        engines_.push_back(std::mt19937_64(rnd()));

    if (is_selftest)
        train_and_test_and_holdout_parody();

    /*

    print_log("1");

    for (size_t j = 0; j < 10; j++)
    {
        auto kv = rand_element(FILTER_GROW);
        std::string s =  std::to_string(j) + ": " +  std::to_string(kv.first) + " : " +  std::to_string(kv.second);
        print_log(s);
    }

    print_log("2");

    exit(0);

    */

    /*
    for (auto &x : FILTER_GROW)
    {
         auto kv = rand_element(FILTER_GROW, p_id);
         std::string s =  std::to_string(j) + ": " +  std::to_string(kv.first) + " : " +  std::to_string(kv.second);
         print_log(s);
         // std::cout << " key : " << x.first << " value : " << x.second << std::endl;
    }

    exit(0);
    */


    std::cout << "start main procedure..." << std::endl;
    auto start_0 = std::chrono::high_resolution_clock::now();

    size_t GLOBAL_ID_LEAF = 1;

    std::cout << "start loading TEST file ..." << std::endl;
    load_test(TEST_FILE);
    std::cout << "finish loading the " << make_colored("", "92", 1) << "TEST" << make_colored("", "92", 2) << " file" << std::endl;

    if (BOOST > 1)
    {
        std::cout << "start loading HOLDOUT file ..." << std::endl;
        load_holdout(HOLDOUT_FILE);
        std::cout << "finish loading the " << make_colored("", "92", 1) << "HOLDOUT" << make_colored("", "92", 2) << " file" << std::endl;
    }

    T_MAT_D res_test;
    T_MAT_D res_holdout;

    hash_holdout_col.clear();
    hash_test_col.clear();

    auto start_1 = std::chrono::high_resolution_clock::now();
    size_t epoch = 1;

    std::unordered_map<std::string, uint32_t> map_stop_line;

    double LOSS_MAX = LOWEST_DOUBLE;
    double LOSS_MIN = MAX_DOUBLE;

    double LOSS_GLOBAL = 0.0;
    size_t n_loss_global = 0;

    double LOSS_FIND = 0.0;
    size_t n_loss_find = 0;


    double W_MAX = LOWEST_DOUBLE;
    double W_MIN = MAX_DOUBLE;

    double W_GLOBAL = 0.0;
    size_t n_w_global = 0;

    double W_FIND = 0.0;
    size_t n_w_find = 0;

    for (;;)
    {
        if (load_command(COMMAND_FILE) == 1) break;

        auto uptime = std::round(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_1).count() * 0.001);

        std::cout << print_now() << std::endl << "UPTIME: " << print_sec(uptime) << std::endl;

        std::cout << "start loading TRAIN file ..." << std::endl;

        if (is_selftest)
            map_stop_line.clear();

        if (LOAD_BIG)
            load_big_train_and_control(TRAIN_FILE, map_stop_line);
        else
            load_train_and_control(TRAIN_FILE);

        std::cout << "finish loading the " << make_colored("", "92", 1) << "TRAIN / CONTROL" << make_colored("", "92", 2) << " file" << std::endl;

        global_uses_train_lines.clear();  

        auto div2 = split_train_and_control();
        auto end_ch = train_and_control.size() - div2;

        T_VEC_SIZE train_last_id(end_ch);

        std::copy(train_and_control_id.begin() + div2, train_and_control_id.end(), train_last_id.begin());

        scale_epoch = prepare_scale(scale);

        std::cout << "epoch " << make_colored("", "72", 1) << epoch << make_colored("", "72", 2) << "/" << make_colored("", "72", 1) << EPOCH << make_colored("", "72", 2) << std::endl << "grow..." << std::endl;

        T_FOREST Forest_LOCAL;

        size_t dog = 0;

        for (;;)
        {
            if (dog > DOG) break;

            T_FOREST Forest(N_FOREST);

            {

            size_t n_threads_ = n_threads;

            if (n_threads > N_FOREST)
                n_threads_ = N_FOREST;

            T_VEC_32 limits = bounds(n_threads_, N_FOREST);

            std::vector<std::thread> th;

            for (size_t id_ = 0; id_ < n_threads_; id_++)
                th.push_back(std::thread(grow, limits[id_], limits[id_ + 1] - 1, std::ref(Forest), std::ref(train), std::ref(train_last_el), std::ref(train_last_id), true, id_ + 1));

            for (auto & t : th) t.join();

            }

            auto predict_res = predict_(Forest, 2);

            auto loss_new = prepare_loss(train_last_el, predict_res.first);

            if ( (MIN_OR_MAX == 1) ? (loss_new > BARRIER_LOSS) : (loss_new < BARRIER_LOSS) )
            {
                std::cout << std::endl <<
                          "-- loss of self (train):"
                          << make_colored("", "36", 1) << print_loss(loss_new) << make_colored("", "36", 2)
                          << make_colored("", "31", 1) << " <----- pass." << make_colored("", "31", 2)
                          << std::endl;
                dog++;

                continue;
            }

            if ( (MIN_OR_MAX == 1) ? (predict_res.second > BARRIER_W) : (predict_res.second < BARRIER_W) )
            {
                std::cout << std::endl <<
                          "-- w of self (train):   "
                          << make_colored("", "36", 1) << predict_res.second << make_colored("", "36", 2)
                          << make_colored("", "31", 1) << " <----- pass." << make_colored("", "31", 2)
                          << std::endl;
                dog++;

                continue;
            }

            size_t nn_F = 0;
            for (auto & T : Forest)
            {
                for (auto & L : T)
                {
                    L.id = GLOBAL_ID_LEAF++;
                    if (L.flag == 2) nn_F++;
                }
            }

            dog = 0;

            std::cout << std::endl <<
                      "-- loss of self (train):"
                      << make_colored("", "90", 1) << print_loss(loss_new) << make_colored("", "90", 2) << std::endl <<
                      "   w of self (train):   "
                      << make_colored("", "90", 1) << predict_res.second << make_colored("", "90", 2) << std::endl <<
                      "   forest have          " << make_colored("", "90", 1) << nn_F << make_colored("", "90", 2) << " leafs" << std::endl;

            Forest_LOCAL = Forest;

            break;

        }

        if (dog > DOG) continue;

        if (!NO_CONTROL)
        {

            std::cout << std::endl << "making control predict..." << std::endl << std::endl;

/*
            {

            MAT_ID.resize(control.size());

            auto res_t = predict_(Forest_LOCAL, 1);

            std::unordered_map<size_t, size_t> hash_p, hash_n;

            for (size_t i = 0; i < control.size(); i++)
            {
                if (IsEqual(res_t[i], MAGIC_1)) continue;

                if (IsEqualE(control_last_el[i], res_t[i]))
                {
                    for (const auto m : MAT_ID[i]) hash_p[m]++;
                }
                else
                {
                    for (const auto m : MAT_ID[i]) hash_n[m]++;
                }
            }

            auto sz_f = Forest_LOCAL.size();
            T_VEC_D mark(sz_f);

            for (size_t j = 0; j < sz_f; j++)
            {
                mark[j] = 0.0;
                size_t nn = 0;

                for (auto & el : Forest_LOCAL[j])
                    if (el.flag == 2)
                    {
                        mark[j] += confidence(hash_p[el.id], hash_n[el.id]);
                        nn++;
                    }

                if (nn > 0) mark[j] /= (double) nn;

                // mark[j] += (double) nn * 0.04;
            }

            auto o = sort_permutation(mark, cmp_double2);

            Forest_LOCAL = apply_permutation(Forest_LOCAL, o);

            Forest_LOCAL.resize(sz_f * 0.1);

            MAT_ID.clear();

            }
*/

            auto predict_res_t = predict_(Forest_LOCAL, 1);
            auto loss_ = prepare_loss(control_last_el, predict_res_t.first);

            if (IsEqual(loss_, MAGIC_1))
            {
                std::cout << make_colored("", "31", 1) << "<----- pass." << make_colored("", "31", 2)
                          << std::endl;

                continue;
            }

            LOSS_FIND += loss_;
            n_loss_find++;

            LOSS_MAX = std::max(LOSS_MAX, loss_);
            LOSS_MIN = std::min(LOSS_MIN, loss_);

            if (!IsEqual(LOSS_MIN, MAX_DOUBLE))
            std::cout << "MIN loss (control):     "  << make_colored("", "90", 1) << print_loss(LOSS_MIN) << make_colored("", "90", 2) << std::endl;

            std::cout << "average loss (control): "  << make_colored("", "35", 1) << print_loss(LOSS_FIND / (double) n_loss_find) << make_colored("", "35", 2) <<
            " N = " << make_colored("", "90", 1) << n_loss_find << make_colored("", "90", 2) << std::endl;

            if (!IsEqual(LOSS_MAX, LOWEST_DOUBLE))
            std::cout << "MAX loss (control):     "  << make_colored("", "90", 1) << print_loss(LOSS_MAX) << make_colored("", "90", 2) << std::endl << std::endl;


            W_FIND += predict_res_t.second;
            n_w_find++;

            W_MAX = std::max(W_MAX, predict_res_t.second);
            W_MIN = std::min(W_MIN, predict_res_t.second);

            if (!IsEqual(W_MIN, MAX_DOUBLE))
            std::cout << "MIN w (control):        "  << make_colored("", "90", 1) << W_MIN << make_colored("", "90", 2) << std::endl;

            std::cout << "average w (control):    "  << make_colored("", "35", 1) << W_FIND / (double) n_w_find << make_colored("", "35", 2) <<
            " N = " << make_colored("", "90", 1) << n_w_find << make_colored("", "90", 2) << std::endl;

            if (!IsEqual(W_MAX, LOWEST_DOUBLE))
            std::cout << "MAX w (control):        "  << make_colored("", "90", 1) << W_MAX << make_colored("", "90", 2) << std::endl;


            if ( (MIN_OR_MAX == 1) ? (loss_ > BARRIER_LOSS) : (loss_ < BARRIER_LOSS) )
            {
                std::cout << std::endl
                          << "-- probe control loss:  "
                          << make_colored("", "33", 1) << print_loss(loss_) << make_colored("", "33", 2)
                          << make_colored("", "31", 1) << " <----- pass." << make_colored("", "31", 2) << std::endl << std::endl;

                continue;
            }


            if ( (MIN_OR_MAX == 1) ? (predict_res_t.second > BARRIER_W) : (predict_res_t.second < BARRIER_W) )
            {
                std::cout << std::endl
                          << "-- probe control w:     "
                          << make_colored("", "33", 1) << predict_res_t.second << make_colored("", "33", 2)
                          << make_colored("", "31", 1) << " <----- pass." << make_colored("", "31", 2) << std::endl << std::endl;

                continue;
            }


            /*

            if (TREE_MUTATE > 0)
            {
                std::cout << std::endl << "try mutate of control predict..." << std::endl;

                auto loss_BEST = loss_;

                auto best_forest = Forest_LOCAL;

                for (size_t nf = 0; nf < TREE_MUTATE; nf++)
                {

                    auto temp_forest = Forest_LOCAL;

                    size_t ind_tree;

                    size_t NT = truly_rand(1, NODE_MUTATE);
                    for (size_t nt = 0; nt < NT; nt++)
                    {
                        ind_tree = truly_rand(0, temp_forest.size() - 1);

                        auto tree = temp_forest[ind_tree];

                        size_t ind_el = truly_rand(1, tree.size() - 1);

                        auto el = tree[ind_el];

                        if ( (el.level < 2) || (el.flag != 1) ) continue;

                        tree[el.parent].left = el.left;
                        tree[el.parent].right = el.right;

                        tree[el.left].parent = el.parent;
                        tree[el.right].parent = el.parent;

                        temp_forest[ind_tree] = tree;
                    }

                    auto res = predict_(temp_forest, 1);

                    auto loss_new = prepare_loss(control_last_el, res);

                    if ( (MIN_OR_MAX == 1) ? (loss_new > loss_BEST) : (loss_new < loss_BEST) )
                        best_forest= temp_forest;
                    else
                    {
                        res_t = res;
                        loss_BEST = loss_new;
                    }
                }

                loss_ = loss_BEST;

                Forest_LOCAL = best_forest;
            }

            */

            std::cout << std::endl << "select loss (control):  "  << make_colored("", "90", 1) << print_loss(loss_) << make_colored("", "90", 2) << std::endl;

            logfile << "select loss (control): " << print_loss(loss_) << std::endl;

            LOSS_GLOBAL += loss_;
            n_loss_global++;


            std::cout << "select w (control):     "  << make_colored("", "90", 1) << predict_res_t.second << make_colored("", "90", 2) << std::endl << std::endl;

            logfile << "select w (control):    " << predict_res_t.second << std::endl;

            W_GLOBAL += predict_res_t.second;
            n_w_global++;


            std::cout << "average loss (control): "
                      << make_colored("", "93", 1) << print_loss(LOSS_GLOBAL / (double) n_loss_global) << make_colored("", "93", 2) <<
              " N = " << make_colored("", "90", 1) << n_loss_global << make_colored("", "90", 2) <<
                      std::endl << "average w (control):    "
                      << make_colored("", "93", 1) << W_GLOBAL / (double) n_w_global << make_colored("", "93", 2) << std::endl << std::endl;

            std::cout << "wait for test predict..." << std::endl << std::endl;

            // START ERR ZONE

            M_FOREST M_Forest_ERR;
            T_FOREST Forest_ERR2;

            if (BOOST > 0)
            {
                train_last_id.resize(div2);

                std::copy(train_and_control_id.begin(), train_and_control_id.begin() + div2, train_last_id.begin());

                auto err_last_el = sub_vec_d(control_last_el, predict_res_t.first, ALPHA);

                for (size_t iter = 0; iter < ITER_ERR; iter++)
                {
                    global_uses_train_lines.clear();  

                    T_FOREST Forest_ERR(N_FOREST_ERR);

                    size_t n_threads_ = n_threads;

                    if (n_threads > N_FOREST_ERR)
                        n_threads_ = N_FOREST_ERR;

                    T_VEC_32 limits = bounds(n_threads_, N_FOREST_ERR);

                    std::vector<std::thread> th;

                    for (size_t id_ = 0; id_ < n_threads_; id_++)
                        th.push_back(std::thread(grow, limits[id_], limits[id_ + 1] - 1, std::ref(Forest_ERR), std::ref(control), std::ref(err_last_el), std::ref(train_last_id), false, id_ + 1));

                    for (auto & t : th) t.join();
        
                    M_Forest_ERR.push_back(Forest_ERR);

                    err_last_el = sub_vec_d(err_last_el, predict_(Forest_ERR, 1).first, ALPHA);
                }                 

                if (SERIALIZE)
                {
                    size_t ind = 1;
                    for (auto f : M_Forest_ERR)
                    {
                        std::ofstream outforest_err(FOREST_FILE + "_ERR_FOREST_" + std::to_string(ind++) + "_EPOCH_" + std::to_string(epoch) + ".dot");
                        serialize(f, outforest_err, "ERR");
                        outforest_err.close();
                    }
                }

                if (BOOST > 1)
                {

                global_uses_train_lines.clear();  

                Forest_ERR2.resize(N_FOREST_ERR2);

                auto res_h = predict_(Forest_LOCAL, 3).first;

                for (auto f : M_Forest_ERR)
                    res_h = sum_vec_d(res_h, predict_(f, 3).first, ALPHA); 

                auto err_last_el = sub_vec_d(holdout_last_el, res_h);

                size_t n_threads_ = n_threads;

                if (n_threads > N_FOREST_ERR2)
                    n_threads_ = N_FOREST_ERR2;

                T_VEC_32 limits = bounds(n_threads_, N_FOREST_ERR2);

                std::vector<std::thread> th;

                for (size_t id_ = 0; id_ < n_threads_; id_++)
                    th.push_back(std::thread(grow, limits[id_], limits[id_ + 1] - 1, std::ref(Forest_ERR2), std::ref(holdout), std::ref(err_last_el), std::ref(holdout_id), false, id_ + 1));

                for (auto & t : th) t.join();

                if (SERIALIZE)
                {
                    std::ofstream outforest_err2(FOREST_FILE + "_ERR2_EPOCH_" + std::to_string(epoch) + ".dot");
                    serialize(Forest_ERR2, outforest_err2, "ERR2");
                    outforest_err2.close();
                }

                }
            }
            // END ERR ZONE

            switch (BOOST)
            {
                case 0 :
                    res_test.push_back(predict_(Forest_LOCAL, 0).first); break;

                case 1 :
                    {
                    auto m_res = predict_(Forest_LOCAL, 0).first;

                    for (auto f : M_Forest_ERR)
                        m_res = sum_vec_d(m_res, predict_(f, 0).first, ALPHA); 

                    res_test.push_back(m_res);

                    break;
                    }

                case 2 :
                    {
                    auto m_res = predict_(Forest_LOCAL, 0).first;

                    for (auto f : M_Forest_ERR)
                        m_res = sum_vec_d(m_res, predict_(f, 0).first, ALPHA); 

                    m_res = sum_vec_d(m_res, predict_(Forest_ERR2, 0).first); 

                    res_test.push_back(m_res);

                    break;
                    }
            }

        } // END NO_CONTROL
        else
        {
            std::cout << "wait for test predict..." << std::endl << std::endl;
            res_test.push_back(predict_(Forest_LOCAL, 0).first);
        }

        std::cout << "------------------------------------------------------------" << std::endl;

        if (SERIALIZE)
        {
            std::ofstream outforest_main(FOREST_FILE + "_MAIN_EPOCH_" + std::to_string(epoch) + ".dot");
            serialize(Forest_LOCAL, outforest_main, "MAIN");
            outforest_main.close();
        }

        if (++epoch > EPOCH) break;

        auto end_1 = std::chrono::high_resolution_clock::now();
        double t1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1).count() / epoch;
        auto t2 = std::round((EPOCH - epoch + 1.0) * t1 * 0.001);
        std::cout << "WAIT:   " << print_sec(t2) << std::endl << std::endl;
    }

    T_VEC_D res_test_v(test.size(), MAGIC_1);

    for (size_t x = 0; x < test.size(); x++)
    {
        T_VEC_D el;
        el.reserve(res_test.size());

        for (size_t y = 0; y < res_test.size(); y++)
        {
            if (!IsEqual(res_test[y][x], MAGIC_1)) el.push_back(res_test[y][x]);
        }

        switch (MEAN_MODEL)
        {
            case 0 : res_test_v[x] = res_r(el);          break;
            case 1 : res_test_v[x] = median(el);         break;
            case 2 : res_test_v[x] = shoorygin(el);      break;
            case 3 : res_test_v[x] = geometric_mean(el); break;
        }
     }

    if (BOOST > 1)
    {

    logfile << std::endl << "holdout importance:" << std::endl;

    std::map<size_t, size_t> temp_o(hash_holdout_col.begin(), hash_holdout_col.end());

    logfile << std::endl;

    for (const auto & x : temp_o)
        logfile << "variable " << x.first + 1 << " : " << x.second << std::endl;

    }

    {

    logfile << std::endl << "test importance:" << std::endl;

    std::map<size_t, size_t> temp_o(hash_test_col.begin(), hash_test_col.end());

    logfile << std::endl;

    for (const auto & x : temp_o)
        logfile << "variable " << x.first + 1 << " : " << x.second << std::endl;

    logfile << std::endl;

    }

    logfile << std::endl << "result:" << std::endl << std::endl;

    for (const auto & x : res_test_v)
    {
        if (IsEqual(x, MAGIC_1))
            logfile << "nan" << std::endl;
        else
            logfile << x << std::endl;
    }

    logfile << std::endl;


    if (TEST_ESTIMATE)
    {
        auto loss_test = prepare_loss(test_last_el, res_test_v);

        std::cout << std::endl << "loss (test):            " << make_colored("", "31", 1) << print_loss(loss_test) << make_colored("", "31", 2) << std::endl;
        logfile << "loss (test):" << std::endl << print_loss(loss_test) << std::endl;
    }


    std::ofstream outfile(RESULT_FILE);

    // std::streamsize save_p = std::cout.precision();

    size_t bad_ = 0;
    size_t row = 0;
    for (const auto & e : test)
    {
        for (const auto & x : e)
            outfile << x << ",";

        std::ostringstream s;
        if (IsEqual(res_test_v[row], MAGIC_1))
        {
             bad_++;
             s << "nan";
        }
        else
        {
            if (FIXED_OUTPUT)
                s << std::fixed << std::setprecision(10) << res_test_v[row];
            else
                s << res_test_v[row];
        }

        outfile << s.str() << std::endl;
        row++;
    }

    outfile.close();

    std::cout << std::endl << "<seed>" << seed_ << "</seed>" << std::endl;
    logfile   << std::endl << "<seed>" << seed_ << "</seed>" << std::endl;

    if (row > 0)
    {
        std::cout << std::endl << "nan : " << bad_ << " (" << bad_ * 100.0 / row << "%)" << std::endl;
        logfile << std::endl << "nan : " << bad_ << " (" << bad_ * 100.0 / row << "%)" << std::endl;
    }

    auto end_0 = std::chrono::high_resolution_clock::now();

    std::cout << std::endl << "------------------------------------------------------------" <<

    std::endl << "ALL TIME: " << print_sec(std::chrono::duration_cast<std::chrono::seconds>(end_0 - start_0).count()) <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end_0 - start_0).count() % 1000 << " ms " << std::endl << std::endl;

    logfile   << "ALL TIME: " << print_sec(std::chrono::duration_cast<std::chrono::seconds>(end_0 - start_0).count(), true) <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end_0 - start_0).count() % 1000 << " ms " << std::endl;

    logfile << std::endl << "-end- " << print_now() << " ------------------------>" << std::endl;

    logfile.close();

    return 0;
}

