//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
using namespace std;

typedef double numeric;
typedef std::vector<numeric> numeric_vector;

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
numeric_vector populate(std::string &filename)
{
  ifstream in(filename.c_str());
  numeric_vector values;
  numeric val;
  
  while (!in.eof())
  {
    in >> val;
    values.push_back(val);
  }

  return values;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
numeric variance(numeric_vector &v, numeric first, numeric second)
{
  return abs(v[first] - v[second]);  
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
numeric total_variance_in_range(numeric_vector &v, int from, int to)
{
  numeric n = 0.0;
  
  if (to - from < 1) return 0;
  
  for (int i = from; i < to; ++i)
  {
    n += variance(v, i-1, i);
  }
  
  return n;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void output_variance_vector(numeric_vector &v)
{
  cout << "variance vector: " << endl;
  int j = 1;
  
  for (numeric_vector::iterator i = v.begin(); i != v.end(); ++i, ++j)
  {
    
    cout << j << "\t" << *i << endl;
  }
  cout << endl;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
numeric average_variance(numeric_vector &v, int from, int to)
{
  return total_variance_in_range(v, from, to) / (to-from+1);
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void calculate_terminations(numeric_vector &v)
{
  int lastStop = 0;
  const float epsilon = 0.05;
  const int repeatedSuccess = 0;
  
  for (int i = 1; i < v.size(); ++i)
  {
    numeric average_var_piecewise = average_variance(v, i-ceil(float(i)/10.0), i);
    numeric average_var_total = average_variance(v, 0, i) / ceil(1+(float(i)/2.0) );
    
    std::cout << "ave_var_piece: [" << i-ceil(float(i)/10.0) << ", " << i << "]: ";
    std::cout << average_var_piecewise << endl;
    std::cout << "ave_var_total: [" << 0 << ", " << i << "]: ";
    std::cout << average_var_total << endl;
    std::cout << endl;

    if (average_var_piecewise < average_var_total && 
        lastStop >= repeatedSuccess &&
        average_var_piecewise < epsilon)
    {
      cout << "STOP -- " << i << endl << endl;
    }
    else if (average_var_piecewise < average_var_total) ++lastStop;
    else lastStop = 0;
    
  }
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
int main()
{
  //std::string filename = "errors_for_mysql_training_dataset.txt";
  std::string filename = "errors_for_boost_training_dataset.txt";
  numeric_vector val = populate(filename);

  output_variance_vector(val);
  
  cout << "  total variance: " << total_variance_in_range(val, 0, val.size()) << endl;
  cout << "average variance: " << average_variance(val, 0, val.size()) << endl;
  cout << endl;

  calculate_terminations(val);
  
  return 0;
}
















