# 第二章课后习题

```
#include <iostream>


// practice 1
void p2_1(void)
{
    std::cout << "Jimmy Chen" << std::endl;
    std::cout << "China, Guangdong Province, Huizhou" << std::endl;

    return;
}


// practice 2
void p2_2(void)
{
    int ilong = 0;
    int iyards = 0;
    std::cout << "Input the distance in long:";
    std::cin >> ilong;
    iyards = 220 * ilong;
    std::cout << "the distance in yards is " << iyards << std::endl;

    return;
}


// practice 3
void string1(void)
{
    std::cout << "Three blind mice" << std::endl;

    return;
}

void string2(void)
{
    std::cout << "See how they run" << std::endl;

    return;
}

void p2_3(void)
{
    string1();
    string1();
    string2();
    string2();

    return;
}


// practice 4
void p2_4(void)
{
    int years = 0;
    int months = 0;
    std::cout << "Enter your age:";
    std::cin >> years;

    months = years * 12;
    std::cout << years << " years contain " << months << " monthes!" << std::endl;

    return;
}


// practice 5
double Celsius2Fahrenheit(double Celsius)
{
    return (1.8 * Celsius + 32.0);
}
void p2_5(void)
{
    double Celsius = 0.0;
    double Fahrenheit = 0.0;
    std::cout << "Please enter a Celsius value :";
    std::cin >> Celsius;
    Fahrenheit = Celsius2Fahrenheit(Celsius);

    std::cout << Celsius << " degrees Celsius is " << Fahrenheit << " defrees Fahrenheit." << std::endl;

    return;
}


// practice 6
double LightYears2Astronomical(double LightYears)
{
    return (LightYears * 63240);
}
void p2_6(void)
{
    double lightYears = 0;
    double astronomical = 0;

    std::cout << "Enter the number of light years: ";
    std::cin >> lightYears;

    astronomical = LightYears2Astronomical(lightYears);

    std::cout << lightYears << " light years = " << astronomical << " astronomical units." << std::endl;

    return;
}


// practice 7
void disTime(int hours, int minutes)
{
    std::cout << "Time: " << hours << ":" << minutes << std::endl;

    return;
}
void p2_7(void)
{
    int hours = 0;
    int minutes = 0;
    std::cout << "Enter the number of hours: ";
    std::cin >> hours;
    std::cout << "Enter the number of minutes: ";
    std::cin >> minutes;
    disTime(hours, minutes);

    return;

}


int main(int argc, char **argv)
{
    p2_7();

    getchar();
}

```

# 第三章课后习题

```
#include <iostream>

using std::cout;
using std::cin;
using std::endl;


//practice 1
const int Foot2Inch = 12;
void p3_1(void)
{
    int inch_input = 0;
    int inch_output = 0;
    int foot = 0;

    cout << "Please input you height in inch:___\b\b\b";
    cin >> inch_input;
    foot = inch_input / Foot2Inch;
    inch_output = inch_input % Foot2Inch;

    cout << "Your height in inch was: " << inch_input << "; Your height in foot and inch was: " << foot << " foot " << inch_output << " inch!" << endl;

    return;
}


// practice 2
const double Inch2Meter = 0.254;
const double Kilo2Pound = 2.2;
void p3_2(void)
{
    double height_foot = 0.0;
    double height_inch = 0.0;
    double height_meter = 0.0;
    double weight_pound = 0.0;
    double weight_kilo = 0.0;
    double BMI = 0.0;

    cout << "Enter your height in foot and inch" << endl;
    cout << "First enter the foot: ";
    cin >> height_foot;
    cout << "Second enter the inch: ";
    cin >> height_inch;

    cout << "Enter you weight in pound: ";
    cin >> weight_pound;

    height_meter = (height_foot * Foot2Inch + height_inch) * Inch2Meter;
    weight_kilo = weight_pound / Kilo2Pound;

    BMI = weight_kilo / (height_meter * height_meter);

    cout << "Your BIM is " << BMI << endl;
}


// practice 3
void p3_3(void)
{
    int degrees = 0;
    int minutes = 0;
    int seconds = 0;
    double total = 0.0;

    cout << "Enter a latitude in degrees, minute and seconds:" << endl;
    cout << "First, enter the degrees: ";
    cin >> degrees;
    cout << "Next, enter the minucts of arc: ";
    cin >> minutes;
    cout << "Finally, enter the seconds of arc: ";
    cin >> seconds;

    total = degrees + minutes / 60.0 + seconds / 3600.0;
    cout << degrees << " degrees, " << minutes << " minutes, " << seconds << " seconds = " << total << " degrees." << endl;
}


// practice 4
void p3_4(void)
{
    long long total_seconds = 0;
    int days = 0;
    int hours = 0;
    int minutes = 0;
    int seconds = 0;

    cout << "Enter the number of seconds: ";
    cin >> total_seconds;

    days = total_seconds / (24 * 60 * 60);
    hours = ((total_seconds % (24 * 60 * 60)) / (60 * 60));
    minutes = ((total_seconds % (60 * 60)) / 60);
    seconds = (total_seconds % 60);

    cout << total_seconds << " seconds = " << days << " days, " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds" << endl;
}


// practice 5
void p3_5(void)
{
    long long world_population = 0;
    long long US_population = 0;
    double rate = 0.0;

    cout << "Enter the world's population :";
    cin >> world_population;

    cout << "Enter the population of the US: ";
    cin >> US_population;

    rate = double(US_population) / (world_population);
    cout << "The population of the US is " << rate * 100 << "% of the world population." << endl;
}


// practice 6
void p3_6(void)
{
    double mile = 0.0;
    double gallon = 0.0;
    double mile_per_gallon = 0.0;

    cout << "Enter the distance in mile you drive: ";
    cin >> mile;
    cout << "Enter the comsumption of oil: ";
    cin >> gallon;

    mile_per_gallon = mile / gallon;
    cout << "Average fuel comsuption: " << mile_per_gallon << " mile/gallon" << endl;
}


// practice 7
void p3_7(void)
{
    double fuel_comsuption_eu = 0.0;
    double fuel_comsuption_us = 0.0;

    cout << "Enter the fuel comsuption in Europ standard: ";
    cin >> fuel_comsuption_eu;

    fuel_comsuption_us = fuel_comsuption_eu / 19 * 12.41;
    cout << "the fuel comsuption in US standard is " << fuel_comsuption_us << "/100KM" << endl;
}


int main(int argc, char **argv)
{
    p3_7();

    while (getchar());

    return 0;
}
```

# 第四章课后习题

```
#include <iostream>
#include <string>
#include <array>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::getline;


// practice 1
void p4_1(void)
{
    char first_name[128];
    char last_name[128];
    char letter;
    int age;

    cout << "What is your first name? ";
    cin.getline(first_name, 128);

    cout << "What is your last name? ";
    cin.getline(last_name, 128);

    cout << "What letter grade do you deserve? ";
    cin >> letter;

    cout << "What is your age? ";
    cin >> age;

    cout << "Name: " << last_name << ", " << first_name << endl;
    cout << "Grade: " << char(letter + 1) << endl;
    cout << "Age: " << age << endl;

}


// practice 2
void p4_2(void)
{
    const int ArSize = 20; // 这句可以去掉
    string name;
    string dessert;

    cout << "Enter your name:\n";
    getline(cin, name);

    cout << "Enter your favourite dessert:\n";
    getline(cin, dessert);

    cout << "I have some delicious " << dessert;
    cout << " for you, " << name << ".\n";

    return;
}


// practice 3
void p4_3(void)
{
    char first_name[256];
    char last_name[256];

    cout << "Enter your fitst name:";
    cin.getline(first_name, 256);
//  cin.get(first_name, 256);   // cin.get的参数和cin.getline相同，但是getline会读取换行符并保存到第一个参数中，但是cin.get会将换行符留在输入队列中，所以会导致下一个读取的内容是换行符

    cout << "Enter your last name:";
    cin.getline(last_name, 256);

    cout << "Here's your infomation is a single string: " << last_name << ", " << first_name << endl;

    return;
}


// pracitce 4
void p4_4(void)
{
    string first_name;
    string last_name;

    cout << "Enter your first name: ";
//  cin.getline()  // cin.getline不适用于string类
    getline(cin, first_name);

    cout << "Enter your last name: ";
    getline(cin, last_name);

    cout << "Here's your infomation is a single string: " << last_name << ", " << first_name << endl;
}


// practice 5
typedef struct CandyBar {
    string name;
    double weight;
    int calories;
} ST_CandyBar;
void p4_5(void)
{
    ST_CandyBar candybar = { "Mocha Munch", 2.3, 350 };
    cout << "the infomation of CandyBar, Name: " << candybar.name << ", Weight: " << candybar.weight << ", " << " Calories: " << candybar.calories << "." << endl;

    return;
}


// practice 6
void p4_6(void)
{
    ST_CandyBar candybar[3] = {
        {"Mocha Munch", 2.3, 350},
        {"Banana", 3.5, 400},
        {"HAHAHAHA", 3.0, 370}
    };

    for (int i = 0; i < 3; i++)
    {
        cout << "the infomation of CandyBar, Name: " << candybar[i].name << ", Weight: " << candybar[i].weight << ", " << " Calories: " << candybar[i].calories << "." << endl;
    }
}


// practice 7
typedef struct PIZZA_INFO {
    string name;
    double size;
    double weight;
} ST_PIZZA_INFO;
void p4_7(void)
{
    ST_PIZZA_INFO pizz_info;

    cout << "Enter the pizza company name: ";
    getline(cin, pizz_info.name);

    cout << "Enter the pizza size: ";
    cin >> pizz_info.size;

    cout << "Enter the pizza weight: ";
    cin >> pizz_info.weight;

    cout << "The company name of pizza is " << pizz_info.name << ", the size of the pizza is " << pizz_info.size << ", the weight of the pizza is " << pizz_info.weight << endl;

    return;
}


// practice 8
void p4_8(void)
{
    ST_PIZZA_INFO *pizza_info = new ST_PIZZA_INFO;

    cout << "Enter the pizza size: ";
    cin >> pizza_info->size;
    cin.get();

    cout << "Enter the pizza company name: ";
    getline(cin, pizza_info->name);

    cout << "Enter the pizza weight: ";
    cin >> pizza_info->weight;

    cout << "The company name of pizza is " << pizza_info->name << ", the size of the pizza is " << pizza_info->size << ", the weight of the pizza is " << pizza_info->weight << endl;

    delete pizza_info;

    return;
}


// practice 9
void p4_9(void)
{
    ST_CandyBar *pcandybar = new ST_CandyBar[3]{
        { "Mocha Munch", 2.3, 350 },
        { "Banana", 3.5, 400 },
        { "HAHAHAHA", 3.0, 370 }
    };

    for (int i = 0; i < 3; i++)
    {
        cout << "the infomation of CandyBar, Name: " << pcandybar[i].name << ", Weight: " << pcandybar[i].weight << ", " << " Calories: " << pcandybar[i].calories << "." << endl;
    }

    delete[] pcandybar;
}


// practice 10
void p4_10(void)
{
    std::array<double, 3> time;
    double avg_time = 0.0;

    printf("Enter three results time of runing 40 meters: ");
    cin >> time[0];
    cin >> time[1];
    cin >> time[2];

    avg_time = (time[0] + time[1] + time[2]) / 3;
    cout << "Result: " << time[0] << ", " << time[1] << ", " << time[2] << endl;
    cout << "Average result: " << avg_time;

}


int main(int argc, char **argv)
{
    p4_10();

    while (cin.get());

    return 0;
}

```

# 第五章课后习题

```
#include <iostream>
#include <string>
#include <array>

using std::cin;
using std::cout;
using std::endl;
using std::array;
using std::string;
using std::getline;


// practice 1
void p5_1(void)
{
    int number1 = 0;
    int number2 = 0;
    int sum = 0;

    cout << "Enter the first number: ";
    cin >> number1;

    cout << "Enter the second number: ";
    cin >> number2;

    for (int i = number1; i <= number2; i++)
    {
        sum += i;
    }

    cout << "The sum of number " << number1 << " to " << number2 << ", sum = " << sum << endl;
}


// practice 2
const int ArSize = 100;
void p5_2(void)
{
    array<long double, ArSize+1> factorials;
    factorials[0] = factorials[1] = 1.0;
    for (int i = 2; i <= ArSize; i++)
    {
        factorials[i] = i * factorials[i - 1];
    }
    for (int i = 0; i <= ArSize; i++)
    {
        cout << i << "! = " << factorials[i] << endl;
    }

    return;
}


// practice 3
void p5_3(void)
{
    int sum = 0;
    int num = 0;

    while (1)
    {
        cout << "Enter a number( 0 to exit): ";
        cin >> num;
        if (num == 0)
        {
            break;
        }
        sum += num;
        cout << "Until now, the sum of the number you inputed is " << sum << endl;
    }

    cout << "Done." << endl;

    return;
}


// practice 4
void p5_4(void)
{
    double Daphne_account = 100;
    double Cleo_account = 100;
    int years = 0;

    while (Daphne_account >= Cleo_account)
    {
        years++;
        Cleo_account = Cleo_account + Cleo_account * 0.05;
        Daphne_account += 10;
    }

    cout << "After " << years << " years. " << "Cleo account which is " << Cleo_account << " will more than Daphne account which is " << Daphne_account << "." << endl;
}


// practie 5
void p5_5(void)
{
    string month[12] = { "January", "February", "March", "April", "May", "June", "July", "August", "Septempber", "October", "November", "December" };
    int sell[12];
    int total_sales = 0;

    cout << "Enter the sales of book <C++ for Fools> each month: " << endl;
    for (int i = 0; i < 12; i++)
    {
        cout << month[i] << ": ";
        cin >> sell[i];
        total_sales += sell[i];
    }

    cout << "The total sales is " << total_sales << endl;
    for (int i = 0; i < 12; i++)
    {
        cout << month[i] << ": " << sell[i] << endl;
    }

    return;
}


// practice 6
void p5_6(void)
{
    string month[12] = { "January", "February", "March", "April", "May", "June", "July", "August", "Septempber", "October", "November", "December" };
    unsigned int sales[3][12] = { 0 };
    unsigned int total_sales[3] = { 0 };

    for (int i = 0; i < 3; i++)
    {
        cout << "Enter " << i + 1 << " year(s) sales of book <C++ for Fools> each month: " << endl;
        for (int j = 0; j < 12; j++)
        {
            cout << month[j] << ": ";
            cin >> sales[i][j];
            total_sales[i] += sales[i][j];
        }
    }

    for (int i = 0; i < 3; i++)
    {
        cout << i + 1 << " year(s) total sales is " << total_sales[i] << endl;
    }

    cout << "Three years total sales is " << total_sales[0] + total_sales[1] + total_sales[2] << endl;
}


//practice7
struct car
{
    string company;
    int pro_year;
};
void p5_7(void)
{
    unsigned int size = 0;
    cout << "How many cars do you wish to catalog? ";
    cin >> size;
    cin.get();

    struct car *pcar = new struct car[size];
    for (unsigned int i = 0; i < size; i++)
    {
        cout << "Car #" << i + 1 << ":" << endl;
        cout << "Please enter the make: ";
        getline(cin, pcar[i].company);

        cout << "Please enter the year make: ";
        cin >> pcar[i].pro_year;
        cin.get();
    }

    cout << "Here is your collection:" << endl;
    for (unsigned int i = 0; i < size; i++)
    {
        cout << pcar[i].pro_year << " " << pcar[i].company << endl;
    }
}


// practice 8
void p5_8(void)
{
    unsigned int n_word = 0;
    char input[128];

    cout << "Enter words (to stop, type the word done):" << endl;
    while (cin >> input)
    {
        if (strcmp(input, "done"))
        {
            n_word++;
        }
        else
            break;
    }

    cout << "You entered a total of " << n_word << " words." << endl;
}


// practice 9
void p5_9(void)
{
    unsigned int n_word = 0;
    string input;

    cout << "Enter words (to stop, type the word done):" << endl;
    while (cin >> input)
    {
        if (input != "done")
        {
            n_word++;
        }
        else
            break;
    }

    cout << "You entered a total of " << n_word << " words." << endl;
}


// practice 10
void p5_10(void)
{
    unsigned int row = 0;

    cout << "Enter number of rows: ";
    cin >> row;

    for (unsigned int i = 1; i <= row; i++)
    {
        for (unsigned int j = i; j < row; j++)
        {
            cout << ".";
        }
        for (unsigned int j = 0; j < i; j++)
        {
            cout << "*";
        }
        cout << endl;
    }
}


int main(int argc, char **argv)
{
    p5_10();

    while (getchar());

    return 0;

```

# 第六章课后习题

```
#include <iostream>
#include <string>
#include <cctype>
#include <array>
#include <iomanip>
#include <fstream>


using namespace std;


// practice 1
void p6_1(void)
{
    char ch = 0;
    while ((ch = cin.get()) != '@')
    {
        if (isdigit(ch))
        {
            continue;
        } 
        else if (islower(ch))
        {
            cout << (char)toupper(ch);
        }
        else if (isupper(ch))
        {
            cout << (char)tolower(ch);
        }
    }

    cout << "Done!" << endl;
}


// practice 2
void p6_2(void)
{
    const unsigned int SIZE = 10;
    array<double, SIZE> donation;
    unsigned int enter = 0;
    double total_value = 0.0;
    double avg = 0.0;
    unsigned int large_avg = 0;

    cout << "Please enter up to ten double value, Non-digital to exit: " << endl;
    while (enter < 10 && (cin >> donation[enter]))
    {
        total_value += donation[enter];
        enter++;
    }

    avg = total_value / enter;
    for (unsigned int i = 0; i < enter; i++)
    {
        if (donation[i] > avg)
        {
            large_avg++;
        }
    }

    cout << "The average value is " << avg << ", and there are " << large_avg << " double value large than agerage value!" << endl;
}


void p6_3(void)
{
    char ch = 0;

    cout << "Please enter one of the following choices: " << endl;
    cout.flags(ios::left);
    cout << setw(20) << "c) carnivore" << "p) pianist" << endl;
    cout << setw(20) << "t) tree" << "g) game" << endl;
    bool exit = false;

    while (cin >> ch && !exit)
    {
        switch (ch)
        {
        case 'c':
            cout << "Tyrannosaurus rex is a carnivore." << endl;
            exit = true;
            break;

        case 'p':
            cout << "Langlang is a pianist." << endl;
            exit = true;
            break;

        case 't':
            cout << "A maple is a tree." << endl;
            exit = true;
            break;

        case 'g':
            cout << "Dota2 is a game." << endl;
            exit = true;
            break;

        default:
            cout << "Please enter a c, p, t or g: ";
            break;
        }
    }
}


// practice 4
const unsigned int strsize = 50;
struct bop
{
    char fullname[strsize];     // real name
    char title[strsize];        // job title
    char bopname[strsize];      // secret BOP name
    int preference;             // 0 = fullname, 1 = title, 2 = bopname
};
void display_by_name(const struct bop *bopArray, unsigned int size)
{
    for (size_t i = 0; i < size; i++)
    {
        cout << bopArray[i].fullname << endl;
    }
}
void display_by_title(const struct bop *bopArray, unsigned int size)
{
    for (size_t i = 0; i < size; i++)
    {
        cout << bopArray[i].title << endl;
    }
}
void display_by_bopname(const struct bop *bopArray, unsigned int size)
{
    for (size_t i = 0; i < size; i++)
    {
        cout << bopArray[i].bopname << endl;
    }
}
void display_by_preference(const struct bop *bopArray, unsigned int size)
{
    for (size_t i = 0; i < size; i++)
    {
        if (bopArray[i].preference == 0)
        {
            cout << bopArray[i].fullname << endl;
        }
        else if (bopArray[i].preference == 1)
        {
            cout << bopArray[i].title << endl;
        }
        else if (bopArray[i].preference == 2)
        {
            cout << bopArray[i].bopname << endl;
        }
    }
}
void p6_4(void)
{
    const struct bop bopArray[5] = {
        {"Wimp Macho", "YYY", "Y----", 0},
        {"XXXXXXXX", "2XXXX", "3XXXXX", 1},
        {"AAAAAAAA", "2AAAA", "3AAAAA", 2},
        {"BBBBBBBB", "2BBBB", "3BBBBB", 0},
        {"CCCCCCCC", "4CCCC", "3CCCCC", 1}
    };
    char choice = 0;

    cout << "Benevolent Order of Proframers Report" << endl;
    cout << left << setw(30) << "a. display by name" << "b. display by title" << endl;
    cout << left << setw(30) << "c. display by bopname" << "d. display by preference" << endl;
    cout << left << setw(30) << "q. quit" << endl;
    cout << "Enter your choice:";

    while (cin >> choice)
    {
        if (choice == 'q')
        {
            break;
        }
        switch (choice)
        {
        case 'a':
            display_by_name(bopArray, 5);
            break;

        case 'b':
            display_by_title(bopArray, 5);
            break;

        case 'c':
            display_by_bopname(bopArray, 5);
            break;

        case 'd':
            display_by_preference(bopArray, 5);
            break;

        default:
            break;
        }
        cout << "Next choice:";
    }
    cout << "Bye!" << endl;
    return;
}


// practice 5
const double RATE1 = 0.1;
const double RATE2 = 0.15;
const double RATE3 = 0.2;
void p6_5(void)
{
    double income = 0.0;
    double tax = 0.0;

    cout << "Please enter your income:";
    while ((cin >> income) && (income > 0))
    {
        if (income <= 5000)
        {
            tax = 0.0;
        }
        else if (5000 < income && income <= 15000)
        {
            tax = (income - 5000) * RATE1;
        }
        else if (15000 < income && income <= 35000)
        {
            tax = (15000 - 5000) * RATE1 + (income - 15000) * RATE2;
        }
        else
        {
            tax = (15000 - 5000) * RATE1 + (35000 - 15000) * RATE2 + (income - 35000) * RATE3;
        }

        cout << "income = " << income << ", tax = " << tax << endl;
    }

    return;
}


// practice 6
struct pat_info
{
    string name;
    double amount;
};
void p6_6(void)
{
    unsigned int contributors = 0;
    unsigned int tmp = 0;

    cout << "Enter the number of contributors:";
    cin >> contributors;
    cin.get();

    struct pat_info *pContributors = new struct pat_info[contributors];

    for (size_t i = 0; i < contributors; i++)
    {
        cout << "Enter the name of " << i + 1 << " contributor: ";
        getline(cin, pContributors[i].name);

        cout << "Enter the amount of donation: ";
        cin >> pContributors[i].amount;
        cin.get();
    }

    cout << "Grand Pators:" << endl;
    for (size_t i = 0; i < contributors; i++)
    {       
        if (pContributors[i].amount > 10000)
        {
            cout << "Contributor name: " << pContributors[i].name << endl;
            cout << "Contrubutor amount: " << pContributors[i].amount << endl;
            tmp++;
        }
    }
    if (tmp == 0)
    {
        cout << "none" << endl;
    }

    tmp = 0;
    cout << "Pators:" << endl;
    for (size_t i = 0; i < contributors; i++)
    {
        if (pContributors[i].amount < 10000)
        {
            cout << "Contributor name: " << pContributors[i].name << endl;
            cout << "Contrubutor amount: " << pContributors[i].amount << endl;
        }
    }
    if (tmp == 0)
    {
        cout << "none" << endl;
    }

    return;
}


// practice 7
void p6_7(void)
{
    unsigned int vowels = 0;
    unsigned int consonants = 0;
    unsigned int others = 0;
    string input;

    cout << "Enter words (q to quit): " << endl;

    while ((cin >> input))
    {
        if (input.length() == 1 && input[0] == 'q')
        {
            break;
        }

        if (isalpha(input[0]))
        {
            if (input[0] == 'a' || input[0] == 'e' || input[0] == 'i' || input[0] == 'o' || input[0] == 'u')
            {
                vowels++;
            }
            else
                consonants++;
        }
        else
            others++;
    }

    cout << vowels << " words beginning with vowels" << endl;
    cout << consonants << " words beginning with consonants" << endl;
    cout << others << " otners" << endl;

    return;
}


// practice 8
void p6_8(void)
{
    string FileName;
    ifstream inFile;
    unsigned int num = 0;
    char ch = 0;

    cout << "Enter the file name:";
    getline(cin, FileName);

    inFile.open(FileName.c_str());

    while ((ch = inFile.get()) != EOF)
    {
        num++;
    }

    cout << "There are " << num << " characters in " << FileName << " file." << endl;
}


// practice 9
void p6_9(void)
{
    unsigned int contributors = 0;
    unsigned int tmp = 0;
    string FileName;
    ifstream inFile;


    cout << "Enter the file name:";
    getline(cin, FileName);
    inFile.open(FileName.c_str());
    inFile >> contributors;
    inFile.get();

    struct pat_info *pContributors = new struct pat_info[contributors];

    for (size_t i = 0; i < contributors; i++)
    {
//      cout << "Enter the name of " << i + 1 << " contributor: ";
        getline(inFile, pContributors[i].name);

//      cout << "Enter the amount of donation: ";
        inFile >> pContributors[i].amount;
        inFile.get();
    }

    cout << "Grand Pators:" << endl;
    for (size_t i = 0; i < contributors; i++)
    {
        if (pContributors[i].amount > 10000)
        {
            cout << "Contributor name: " << pContributors[i].name << endl;
            cout << "Contrubutor amount: " << pContributors[i].amount << endl;
            tmp++;
        }
    }
    if (tmp == 0)
    {
        cout << "none" << endl;
    }

    tmp = 0;
    cout << "Pators:" << endl;
    for (size_t i = 0; i < contributors; i++)
    {
        if (pContributors[i].amount < 10000)
        {
            cout << "Contributor name: " << pContributors[i].name << endl;
            cout << "Contrubutor amount: " << pContributors[i].amount << endl;
            tmp++;
        }
    }
    if (tmp == 0)
    {
        cout << "none" << endl;
    }

    return;
}


int main(int argc, char **argv)
{
    p6_9();

    while (cin.get());

    return 0;
}

```

# 第七章课后习题

```
#include <iostream>
#include <string>
#include <cctype>


using namespace std;



// practice 1
void p7_1(void)
{
    double x, y;
    double avg = 0;

    cout << "Enter two numbers: ";
    cin >> x >> y;
    while (x != 0 && y != 0)
    {
        avg = 2.0 * x * y / (x + y);
        cout << "The average of " << x << " and " << y << " is " << avg << endl;

        cout << "Enter the next two numbers: ";
        cin >> x >> y;
    }
}


// practice 2
void GetInput(double *grade, unsigned int *number)
{
    cout << "You can enter up to 10 grades( -1 to quit): " << endl;
    while (cin >> grade[(*number)++])
    {
        if (grade[*number - 1] == -1)
        {
            break;
        }
    }
    (*number)--;
}
void PrintArray(const double *grage, const unsigned int number)
{
    cout << "The grade is: " << endl;

    for (unsigned int i = 0; i < number; i++)
    {
        cout << grage[i] << " ";
    }

    cout << endl;
}
void CalAvg(const double *grade, const unsigned int number)
{
    double total = 0.0;
    cout << "The average is :";
    for (unsigned int i = 0; i < number; i++)
    {
        total += grade[i];
    }
    cout << total / number << endl;
}
void p7_2(void)
{
    double grade[10];
    unsigned int enter = 0;

    GetInput(grade, &enter);
    PrintArray(grade, enter);
    CalAvg(grade, enter);

    return;
}


// practice 3
struct box
{
    char maker[40];
    float height;
    float width;
    float length;
    float volume;
};
void PrintBox(struct box mbox)
{
    cout << "Box maker: " << mbox.maker << endl;
    cout << "Box height: " << mbox.height << endl;
    cout << "Box width: " << mbox.width << endl;
    cout << "Box length: " << mbox.length << endl;
    cout << "Box volume: " << mbox.volume << endl;
}
void CalBoxVolume(struct box *pbox)
{
    pbox->volume = pbox->height * pbox->width * pbox->length;
}
void p7_3(void)
{
    struct box mbox = { "Jimmy Chen", 0.25, 4.0, 1.0, 0.0 };
    PrintBox(mbox);
    CalBoxVolume(&mbox);
    PrintBox(mbox);
}


// practice 4
long double probability(unsigned number, unsigned picks)
{
    long double result = 1.0;
    long double n = 0;
    unsigned p;
    for (n = number, p = picks; p > 0; n--, p--)
    {
        result = result * n / p;
    }

    return result;
}
void p7_4(void)
{
    unsigned field1 = 47;
    unsigned field2 = 27;

    cout << "You have on chance in ";
    cout << probability(47, 5) * probability(27, 1);
    cout << " of winning.\n" << endl;
}


// practice 5
long long factorial(unsigned int number)
{
    if (number == 0 || number == 1)
    {
        return 1;
    }
    else
    {
        return number * factorial(number - 1);
    }
}
void p7_5(void)
{
    long long result = 0;
    unsigned int input = 0;

    cout << "Please enter the number: ";
    cin >> input;

    while (1)
    {
        result = factorial(input);
        cout << "The result of " << input << "! is " << result << "." << endl;

        cout << "Please enter the next number: ";
        cin >> input;
    }
}


// practice 6
unsigned int Fill_array(double double_array[], unsigned int length)
{
    unsigned int n_input = 0;
    cout << "Enter double numbers (non-digital to quit): " << endl;
    for (size_t i = 0; i < length; i++)
    {
        if (cin >> double_array[i])
        {
            n_input++;
        }
        else
        {
            break;
        }
    }

    return n_input;
}
void Show_array(double double_array[], unsigned int length)
{
    cout << "The double array is :" << endl;
    for (size_t i = 0; i < length; i++)
    {
        cout << double_array[i] << " ";
        if ((i + 1) % 5 == 0)
        {
            cout << endl;
        }
    }
    if (length % 5 != 0)
    {
        cout << endl;
    }
}
void Reverse_array(double double_array[], unsigned int length)
{
    cout << "Revert the array" << endl;
    for (size_t i = 0; i < length / 2; i++)
    {
        double tmp = double_array[i];
        double_array[i] = double_array[length - 1 - i];
        double_array[length - 1 - i] = tmp;
    }
}
void p7_6(void)
{
    double double_array[10];
    unsigned int n_input;
    n_input = Fill_array(double_array, 10);
    Show_array(double_array, n_input);
    Reverse_array(double_array, n_input);
    Show_array(double_array, n_input);
}


// practice 7
int fill_array(double *ar_begin, double *ar_end)
{
    double temp = 0.0;
    int i = 0;
    double *ar_tmp = nullptr;
    for (ar_tmp = ar_begin; ar_tmp < ar_end; ar_tmp++)
    {
        cout << "Enter value #" << (i + 1) << ": ";
        cin >> temp;
        if (!cin)
        {
            cin.clear();
            while (cin.get() != '\n')
                continue;
            cout << "Bad input; input preocess terminated.\n";
            break;
        }
        else if (temp < 0)
        {
            break;
        }

        *ar_tmp = temp;
        i++;
    }

    return i;
}
void show_array(const double *ar_begin, const double *ar_end)
{
    const double *ar_tmp = nullptr;
    unsigned int i = 0;
    for (ar_tmp = ar_begin; ar_tmp < ar_end; ar_tmp++)
    {
        cout << "Property #" << (i + 1) << ": $";
        cout << *ar_tmp << endl;
        i++;
    }
}
void revalue(double *ar_begin, double *ar_end, double r)
{
    for (double * ar = ar_begin; ar < ar_end; ar++)
    {
        *ar *= r;
    }
}
void p7_7(void)
{
    double properties[10];

    int size = fill_array(properties, properties + 10);
    show_array(properties, properties + size);
    if (size > 0)
    {
        cout << "Enter revaluation factor: ";
        double factor;
        while (!(cin >> factor))
        {
            cin.clear();
            while (cin.get() != '\n')
                continue;
            cout << "Bad input; Please enter a number: ";
        }
        revalue(properties, properties + size, factor);
        show_array(properties, properties + size);
    }

    cout << "Done.\n";

    return;
}


// practice 8
const int Seasons = 4;
const char *Snames[Seasons] = { "Spring", "Summer", "Fall", "Winter" };
struct cost
{
    double expenses[Seasons];
};
void fill(double *pa)
{
    for (int i = 0; i < Seasons; i++)
    {
        cout << "Enter " << Snames[i] << " expenses: ";
        cin >> pa[i];
    }
}
void show(double *pa)
{
    double total = 0.0;
    cout << "\nEXPENSES\n";
    for (int i = 0; i < Seasons; i++)
    {
        cout << Snames[i] << ": $" << pa[i] << endl;
        total += pa[i];
    }
    cout << "Total Expenses: $" << total << endl;
}
void fill(struct cost *pCost)
{
    for (int i = 0; i < Seasons; i++)
    {
        cout << "Enter " << Snames[i] << " expenses: ";
        cin >> pCost->expenses[i];
    }
}
void show(struct cost *pCost)
{
    double total = 0.0;
    cout << "\nEXPENSE\n";
    for (int i = 0; i < Seasons; i++)
    {
        cout << Snames[i] << ": $" << pCost->expenses[i] << endl;
        total += pCost->expenses[i];
    }
    cout << "Total Expense: $" << total << endl;
}
void p7_8(void)
{
    // situation a
    cout << "Situation a: " << endl;
    double pa[Seasons] = { 0 };
    fill(pa);
    show(pa);

    // situation b
    cout << endl << endl;
    cout << "Situation b: " << endl;
    struct cost *pCost = new struct cost;
    fill(pCost);
    show(pCost);
    delete pCost;
}


// practice 9
const int SLEN = 30;
struct student
{
    char fullname[SLEN];
    char hobby[SLEN];
    int ooplevel;
};
int getinfo(student pa[], int n)
{
    int enter = 0;
    for (int i = 0; i < n; i++)
    {
        cout << "Enter the infomation of student #" << i+1 << endl;
        cout << "Enter full name (blank line to quit): ";
        cin.getline(pa[i].fullname, SLEN);
        cout << "Enter hobby: ";
        cin.getline(pa[i].hobby, SLEN);
        cout << "Enter ooplevel: ";
        cin >> pa[i].ooplevel;
        while (cin.get() != '\n')
            continue;
        enter++;
    }
    return enter;
}
void display1(student st)
{
    cout << "Using display1(student st): " << endl;
    cout << "Full name: " << st.fullname << endl;
    cout << "Hobby: " << st.hobby << endl;
    cout << "Ooplevel: " << st.ooplevel << endl;
}
void display2(const student *st)
{
    cout << "Using display2(const student *st)" << endl;
    cout << "Full name: " << st->fullname << endl;
    cout << "Hobby: " << st->hobby << endl;
    cout << "Ooplevel: " << st->ooplevel << endl;
}
void display3(const student pa[], int n)
{
    cout << "Using display3(const student pa[], int n)" << endl;;
    for (int i = 0; i < n; i++)
    {
        cout << "Infomation of student #" << i + 1 << ": " << endl;
        cout << "Full name: " << pa[i].fullname << endl;
        cout << "Hobby: " << pa[i].hobby << endl;
        cout << "Ooplevel: " << pa[i].ooplevel << endl;
    }
}
void p7_9(void)
{
    cout << "Enter class size: ";
    int class_size;
    cin >> class_size;
    while (cin.get() != '\n')
        continue;

    student *ptr_stu = new student[class_size];
    int entered = getinfo(ptr_stu, class_size);
    for (int i = 0; i < entered; i++)
    {
        display1(ptr_stu[i]);
        display2(&ptr_stu[i]);
    }
    display3(ptr_stu, entered);

    delete[] ptr_stu;
    cout << "Done.\n";
    return;
}


// practice 10
double add(double x, double y)
{
    return x + y;
}
double mul(double x, double y)
{
    return x * y;
}
double calculate(double x, double y, double(*fun)(double, double))
{
    return fun(x, y);
}
void p7_10(void)
{
    double x = 0.0;
    double y = 0.0;

    cout << "Enter two double number: ";
    while (cin >> x >> y)
    {
        cout << "Call add, the result of " << x << " and " << y << " is " << calculate(x, y, add) << endl;
        cout << "Call mul, the result of " << x << " abd " << y << " is " << calculate(x, y, mul) << endl;

        cout << "Enter next two double number: ";
    }
}


int main(int argc, char **argv)
{
    p7_10();

    while (getchar());

    return 0;
}

```

# 第八章课后习题

```
#include <iostream>
#include <string>
#include <cctype>
#include <cstring>


using namespace std;



// practice 1
void fun_of_p8_1(char *str, int print_times = 0)
{
    // 这里也将print_times也打印出来，方便演示
    cout << "String: " << str << " , print_times: " << print_times << endl;
    if (print_times > 1)
    {
        print_times--;
        fun_of_p8_1(str, print_times);
    }
    return;
}
void p8_1(void)
{
    char str[128];
    int print_times = 0;
    cout << "Enter the string you want to print: ";
    cin.getline(str, 128);
    cout << "Enter the times you want to print: ";
    cin >> print_times;

    // 测试传入两个参数
    cout << "Test fun which has two prematers: " << endl;
    fun_of_p8_1(str, print_times);

    // 测试传入一个参数
    cout << "Test fun which has one premater: " << endl;
    fun_of_p8_1(str);
}


// practice 2
struct CandyBar {
    char company[128];
    double weight;
    int heat;
};
void print_CandyBar(const struct CandyBar &candybar)
{
    cout << "CandyBar company: " << candybar.company << endl;
    cout << "CandyBar weight: " << candybar.weight << endl;
    cout << "CandyBar heat: " << candybar.heat << endl;
}
void fill_CandyBar(struct CandyBar &candybar, char *company, double weight, int heat)
{
    /*
    如果使用strcpy遇到如下错误：
    error C4996: 'strcpy': This function or variable may be unsafe. Consider using strcpy_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    1>c:\program files (x86)\windows kits\10\include\10.0.15063.0\ucrt\string.h(136): note: 参见“strcpy”的声明

    可以这样子解决：右键工程名-->属性-->C/C++-->预处理器-->预处理器定义，编辑右边输入框加入：_CRT_SECURE_NO_WARNINGS
    */
    strcpy(candybar.company, company);
    candybar.weight = weight;
    candybar.heat = heat;
}
void p8_2(void)
{
    struct CandyBar candybar = { "" };
    char company[128];
    double weight = 0.0;
    int heat = 0;

    cout << "Enter the company of candybar: ";
    cin.getline(company, 128);

    cout << "Enter the weight of candybar: ";
    cin >> weight;

    cout << "Enter the heat of candybar: ";
    cin >> heat;

    fill_CandyBar(candybar, company, weight, heat);
    print_CandyBar(candybar);

    return;
}


// practice 3
void p8_3(void)
{
    string input;

    cout << "Enter a string (q to quit): ";
    getline(cin, input);

    while (input[0] != 'q')
    {
        for (size_t i = 0; i < input.length(); i++)
        {
            input[i] = toupper(input[i]);
        }
        cout << input << endl;

        cout << "Next string (q to quit): ";
        getline(cin, input);
    }

}


// practice 4
struct stringy {
    char *str;
    int ct;
};
void set(struct stringy &in_stringy, char * in_string)
{
    int string_length = strlen(in_string);
    in_stringy.str = new char(string_length + 1);
    strcpy(in_stringy.str, in_string);
    in_stringy.ct = string_length;
}
void show(const struct stringy &in_stringy, int print_times = 1)
{
    for (int i = 0; i < print_times; i++)
    {
        cout << "member string of struct stringy: " << in_stringy.str << endl;
    }
}
void show(const char * str, int print_times = 1)
{
    for (int i = 0; i < print_times; i++)
    {
        cout << "Print char string: " << str << endl;
    }
}
int p8_4(void)
{
    stringy beany;
    char testing[] = "Reality isn't what it used to be.";

    set(beany, testing);
    show(beany);
    show(beany, 2);
    testing[0] = 'D';
    testing[1] = 'u';
    show(testing);
    show(testing, 3);
    show("Done!");

    return 0;
}


// practice 5
template <typename T>
T max5(T in_array[])
{
    T max = in_array[0];
    for (size_t i = 0; i < 5; i++)
    {
        if (max < in_array[i])
        {
            max = in_array[i];
        }
    }

    return max;
}
void p8_5(void)
{
    int int_array[5] = {32, 43, 66, 23, 54};
    double double_array[5] = { 32.4, 33.3, 61.3, 646.3, 23.5 };

    int int_array_max = max5(int_array);
    double double_array_max = max5(double_array);

    cout << "max of int array: " << int_array_max << endl;
    cout << "max of double array: " << double_array_max << endl;
}


// practice 6
template <typename T>
T maxn(T in_array[], int array_size)
{
    T max = in_array[0];
    for (int i = 0; i < array_size; i++)
    {
        if (max < in_array[i])
        {
            max = in_array[i];
        }
    }

    return max;
}
// 显示具体化
template <> const char * maxn(const char *in_str[], int n)
{
    const char * str = in_str[0];

    for (int i = 0; i < n; i++)
    {
        if (strlen(str) < strlen(in_str[i]))
        {
            str = in_str[i];
        }
    }

    return str;
}
void p8_6(void)
{
    int int_array[6] = { 43, 235, 54, 232, 123, 65 };
    double double_array[4] = { 32.1, 453.2, 53.3, 67.4 };
    const char * str_array[5] = { "Hello Jimmy!", "Hello World!", "ABCDEFG,HIJKLMN", "Today is a goood day!", "C++ Primer Plus!" };

    int int_max = maxn(int_array, 6);
    double double_max = maxn(double_array, 4);
    const char * length_max_str = maxn(str_array, 5);

    cout << "max of int array: " << int_max << endl;
    cout << "max of double array: " << double_max << endl;
    cout << "max length string of string array: " << length_max_str << endl;
}


// practice 7
template <typename T>
T SumArray(T arr[], int n);

template <typename T>
T SumArray(T * arr[], int n);

struct debts {
    char name[50];
    double amount;
};

int p8_7(void)
{
    int thing[6] = { 13, 31, 103, 301, 310, 130 };
    int int_sum = 0;
    struct debts mr_E[3] = 
    {
        {"Ima Wolfe", 2400.0},
        {"Ura Foxe", 1300.0},
        {"Iby Stout", 1800.0}
    };
    double *pd[3];
    double double_sum = 0.0;

    for (size_t i = 0; i < 3; i++)
    {
        pd[i] = &mr_E[i].amount;
    }

    int_sum = SumArray(thing, 6);
    double_sum = SumArray(pd, 3);

    cout << "Sum of int array: " << int_sum << endl;
    cout << "Sum of double* array: " << double_sum << endl;

    return 0;
}

template <typename T>
T SumArray(T arr[], int n)
{
    // 下面的sum初始化，在便宜的时候会有warning提示，有个简单的方法可以解决，那就是将sum初始化为arr的第一个元素，然后for循环从i=1开始
    T sum = 0.0;

    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }

    return sum;
}

template <typename T>
T SumArray(T * arr[], int n)
{
    T sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += *(arr[i]);
    }

    return sum;
}


// main
int main(int argc, char **argv)
{
    p8_7();


    while (cin.get());
}
```

# 第九章课后习题

```
// chapter9.cpp

#include <iostream>
#include <string>
#include <cstring>
#include <cctype>
#include <new>
#include "sales.h"


using namespace std;


// practice 1
const int Len = 40;
struct golf {
    char fullname[Len];
    int handicap;
};
void setgolf(golf & g, const char * name, int hc)
{
    strcpy(g.fullname, name);
    g.handicap = hc;
}
int setgolf(golf & g)
{
    cout << "Please enter the full name of golf player: ";
    cin.getline(g.fullname, Len);
    if (strcmp(g.fullname, "") == 0)
    {
        return 0;
    }

    cout << "Please enter the hanicap of golf player: ";
    cin >> g.handicap;
    cin.get();

    return 1;
}
void handicap(golf & g, int hc)
{
    g.handicap = hc;
}
void showgolf(const golf & g)
{
    cout << "full name of golf player: " << g.fullname << endl;
    cout << "handicap of golf player: " << g.handicap << endl;
}
void p9_1(void)
{
    golf g[10];
    int n = 0;
    cout << "Enter the information of golf player: " << endl;

    while ((n < 10) && (setgolf(g[n])))
    {
        n++;
        cout << "Next golf player: " << endl;
    }

    cout << "Show all golf player information: " << endl;
    for (int i = 0; i < n; i++)
    {
        showgolf(g[i]);
    }

}


// practice 2
const int ArSize = 10;
void strcount(const string str)
{
    static int total = 0;
    int count = 0;

    cout << "\"" << str.c_str() << "\" contains ";
    while (str[count])
    {
        count++;
    }
    total += count;
    cout << count << " characters\n";
    cout << total << " characters total\n";
}
void p9_2(void)
{
    string input;

    cout << "Enter a line:\n";
    getline(cin, input);
    while (cin)
    {
        strcount(input);
        cout << "Enter next line (empty line to quit): \n";
        getline(cin, input);
        if (input == "")
        {
            break;
        }
    }

    cout << "Bye\n";
    return;
}


// practice 3
struct chaff {
    char dross[20];
    int slag;
};
char buffer[1024];
void p9_3(void)
{
    chaff *pcha = new (buffer) chaff[2];
    char *pc = new char[1024];
    chaff *pcha2 = new (pc) chaff[2];
    char dross[20] = { 0 };
    int slag = 0;

    for (int i = 0; i < 2; i++)
    {
        cout << "Enter dross of #" << i << " chaff: " << endl;
        cin.getline(dross, 20);
        cout << "Enter slag of #" << i << " chaff: " << endl;
        cin >> slag;
        cin.get();

        strcpy(pcha[i].dross, dross);
        strcpy(pcha2[i].dross, dross);
        pcha[i].slag = pcha2[i].slag = slag;
    }

    for (int i = 0; i < 2; i++)
    {
        cout << "staff #" << (i + 1) << ":" << endl;
        cout << "pcha.dross: " << pcha[i].dross << ". pcha.slag: " << pcha[i].slag << endl;
        cout << "pcha2.dross: " << pcha2[i].dross << ". pcha2.slag: " << pcha2[i].slag << endl;
    }

    cout << "address of buffer: " << (void *)buffer << endl;
    cout << "address of pcha: " << pcha << ". address of pcha[0]: " << &pcha[0] << ". address of pcha[1]: " << &pcha[1] << endl;
    cout << "address of pc: " << (void *)pc << endl;
    cout << "address of pcha2:" << pcha2 << ". address of pcha2[0]: " << &pcha2[0] << ". address of pcha2[1]: " << &pcha2[1] << endl;;

    delete[] pc;
}


// practice 4
void p9_4(void)
{
    SALES::Sales sales1;
    SALES::Sales sales2;

    double ar[3] = { 32.1, 23.2, 65.3 };
    SALES::setSales(sales1, ar, 3);
    SALES::setSales(sales2);

    SALES::showSales(sales1);
    SALES::showSales(sales2);
}


int main(int argc, char **argv)
{
    p9_4();

    while (cin.get());

    return 0;
}


// sales.h

#pragma once
#include <iostream>
namespace SALES
{
    const int QUARTERS = 4;
    struct Sales
    {
        double sales[QUARTERS];
        double average;
        double max;
        double min;
    };
    void setSales(Sales & s, const double ar[], int n);
    void setSales(Sales & s);
    void showSales(const Sales & s);
}


// sales.cpp

#include "sales.h"

namespace SALES
{
    using namespace std;

    void setSales(Sales & s, const double ar[], int n)
    {
        double total = 0.0;
        double max = ar[0];
        double min = ar[0];

        for (int i = 0; i < n; i++)
        {
            total += ar[i];
            s.sales[i] = ar[i];
            if (max < ar[i])
            {
                max = ar[i];
            }
            if (min > ar[i])
            {
                min = ar[i];
            }
        }
        for (int i = n; i < QUARTERS; i++)
        {
            s.sales[i] = 0.0;
        }

        s.average = total / n;
        s.max = max;
        s.min = min;
    }

    void setSales(Sales & s)
    {
        double total = 0.0;
        double max = 0.0;
        double min = 0.0;
        double input = 0.0;

        cout << "Enter 4 double number; " << endl;
        for (size_t i = 0; i < QUARTERS; i++)
        {
            cin >> input;
            if (i == 0)
            {
                max = input;
                min = input;
            }
            if (min > input)
            {
                min = input;
            }
            if (max < input)
            {
                max = input;
            }
            total += input;
            s.sales[i] = input;
        }
        s.average = total / QUARTERS;
        s.max = max;
        s.min = min;
    }

    void showSales(const Sales & s)
    {
        cout << "sales: ";
        for (size_t i = 0; i < QUARTERS; i++)
        {
            cout << s.sales[i] << " ";
        }

        cout << endl;
        cout << "average: " << s.average << endl;
        cout << "max: " << s.max << endl;
        cout << "min: " << s.min << endl;
    }
}

```

# 第十章课后习题

```
// chapter10.cpp

#include <iostream>
#include <string>
#include <cstring>
#include <cctype>

#include "Sales.h"
#include "stack.h"
#include "List.h"

using namespace std;

// practice 1
class BankCount {

private:
    string fullname;
    string banknumber;
    long balance;

public:
    BankCount() {}
    BankCount(string fullname, string banknumber, long balance);
    void ShowBankCount();
    int Deposit(long deposit);
    int WithDrawals(long withdrawals);
};
BankCount::BankCount(string fullname, string banknumber, long balance)
{
    this->fullname = fullname;
    this->banknumber = banknumber;
    this->balance = balance;
}
void BankCount::ShowBankCount()
{
    cout << "BankCount full name: " << fullname << endl;
    cout << "BankCount number: " << banknumber << endl;
    cout << "BankCount balance: " << balance << endl;
}
int BankCount::Deposit(long deposit)
{
    this->balance += deposit;
    return 1;
}
int BankCount::WithDrawals(long withdrawals)
{
    this->balance -= withdrawals;
    return 1;
}
void p10_1(void)
{
    BankCount MyBankCount("Jimmy", "1234567", 10000);
    cout << "Original BankCount: " << endl;
    MyBankCount.ShowBankCount();

    cout << endl;
    cout << "After Deposit 5000: " << endl;
    MyBankCount.Deposit(5000);
    MyBankCount.ShowBankCount();

    cout << endl;
    cout << "After withdrawals 5000: " << endl;
    MyBankCount.WithDrawals(5000);
    MyBankCount.ShowBankCount();
}


// practice 2
class Person {
private:
    // static const LIMIT = 25 编译报错，C++不支持默认int
    static const int LIMIT = 25;
    string lname;
    char fname[LIMIT];

public:
    Person() { lname = ""; fname[0] = '\0'; }
    Person(const string & ln, const char * fn = "Heyyou");
    void Show() const;
    void FormalShow() const;
};
Person::Person(const string & ln, const char * fn)
{
    lname = ln;
    strncpy(fname, fn, LIMIT);
}
void Person::Show() const
{
    cout << fname << ", " << lname << endl;
}
void Person::FormalShow() const
{
    cout << lname << ", " << fname << endl;
}
void p10_2(void)
{
    Person one;
    Person two("Smythecraft");
    Person three("Dimwiddy", "Sam");
    one.Show();
    one.FormalShow();

    cout << endl;
    two.Show();
    two.FormalShow();

    cout << endl;
    three.Show();
    three.FormalShow();
}


// practice 3
class golf {
private:
    static const int Len = 40;
    char fullname[Len];
    int handicap;

public:
    golf(const char * name, int hc);
    golf();
    void sethandicap(int hc);
    void showgolf();
};
golf::golf(const char * name, int hc)
{
    strncpy(fullname, name, Len);
    handicap = hc;
}
golf::golf()
{
    char temp[Len] = "";
    int hand = 0;
    cout << "Please enter the full name of golf player: ";
    cin.getline(temp, Len);

    cout << "Please enter the hanicap of golf player: ";
    cin >> hand;
    cin.get();

    *this = golf(temp, hand);
}
void golf::sethandicap(int hc)
{
    handicap = hc;
}
void golf::showgolf()
{
    cout << "golf full name: " << fullname << endl;
    cout << "golf handicap: " << handicap << endl;
}
void p10_3(void)
{
    golf g;
    golf g2("Jimmy", 100);

    g.showgolf();
    g.sethandicap(120);
    g.showgolf();

    g2.showgolf();
    g2.sethandicap(120);
    g2.showgolf();
}


// practice 4
using namespace SALES;
void p10_4(void)
{
    double ar[3] = { 32.1, 23.2, 65.3 };
    Sales sales1(ar, 3);
    Sales sales2;

    sales1.showSales();
    sales2.showSales();
}


// practice 5
void p10_5(void)
{
    Stack stack;
    double total = 0.0;
    customer pop;

    customer customer1 = { "jimmy", 1000 };

    if (stack.push(customer1))
    {
        cout << customer1.fullname << " push." << endl;
    }
    else
    {
        cout << "Stack full." << endl;
    }

    customer customer2 = { "hey", 200.0 };

    if (stack.push(customer2))
    {
        cout << customer2.fullname << " push." << endl;
    }
    else
    {
        cout << "Stack full." << endl;
    }

    if (stack.pop(pop))
    {
        cout << pop.fullname << " pop." << endl;
        total += pop.payment;
    }
    else
        cout << "Stack empty." << endl;


    customer customer3 = { "kitty", 3000.0 };
    if (stack.push(customer3))
        cout << customer3.fullname << " push." << endl;
    else
        cout << "Stack full." << endl;

    if (stack.pop(pop))
    {
        cout << pop.fullname << " pop." << endl;
        total += pop.payment;
    }
    else
        cout << "Stack empty." << endl;

    if (stack.pop(pop))
    {
        cout << pop.fullname << " pop." << endl;
        total += pop.payment;
    }
    else
        cout << "Stack empty." << endl;

    if (stack.pop(pop))
    {
        cout << pop.fullname << " pop." << endl;
        total += pop.payment;
    }
    else
        cout << "Stack empty." << endl;

    cout << "Total paymemt: " << total << endl;
}


// practice 6
class Move
{
private:
    double x;
    double y;

public:
    Move(double a = 0, double b = 0);
    void showmove() const;
    Move add(const Move & m) const;
    void reset(double a = 0, double b = 0);
};
Move::Move(double a, double b)
{
    x = a;
    y = b;
}
void Move::showmove() const
{
    cout << "x = " << x << ", y = " << y << "." << endl;
}
Move Move::add(const Move & m) const
{
    Move temp;
    temp.x = this->x + m.x;
    temp.y = this->y + m.y;

    return temp;
}
void Move::reset(double a, double b)
{
    x = a;
    y = b;
}
void p10_6(void)
{
    Move move1(1.1, 2.2);
    move1.showmove();

    Move move2(3.3, 4.4);
    move2.showmove();

    Move move3 = move1.add(move2);
    move3.showmove();

    move3.reset(0.1, 0.2);
    move3.showmove();
}


// practice 7
class plorg
{
private:
    enum { LEN = 20 };
    char fullname[LEN];
    int CI;

public:
    plorg(char * name = "Plorga", int CI = 50);
    void setCI(int CI);
    void showPlorg();
};
plorg::plorg(char * name, int CI)
{
    strncpy(fullname, name, LEN);
    this->CI = CI;
}
void plorg::setCI(int CI)
{
    this->CI = CI;
}
void plorg::showPlorg()
{
    cout << "plorg full name: " << fullname << endl;
    cout << "plorg CI: " << CI << endl;
}
void p10_7(void)
{
    plorg pl("Jimmy", 100);
    pl.showPlorg();

    plorg pl2;
    pl2.showPlorg();

    pl.setCI(150);
    pl.showPlorg();
}


// practice 8
// 大概这样简单的写一下吧
void p10_8(void)
{
    List list;
    if (list.isempty())
    {
        cout << "List is empty" << endl;
    }

    uItem item1 = 1;
    uItem item2 = 2;
    uItem item3 = 3;

    list.add(item1);
    list.add(item2);
    list.add(item3);

    if (list.isempty())
    {
        cout << "List is empty" << endl;
    }
    else if (list.isfull())
    {
        cout << "List is full" << endl;
    }
    else
    {
        cout << "Some items in the list" << endl;
    }

    list.visit(oringin);
    list.visit(x2);
    list.visit(half);

}


int main(int argc, char **argv)
{
    p10_8();

    while (cin.get());

    return 0;
}


// List.h
#pragma once
#include <iostream>

typedef unsigned long uItem;
using std::cout;
using std::cin;
using std::endl;

class List
{
private:
    enum { MAX = 20 };
    uItem items[MAX];
    int top;

public:
    List() { top = 0; }
    int add(uItem & item);
    bool isempty();
    bool isfull();
    void visit(void(*pf)(uItem &));
};

void oringin(uItem & item);
void x2(uItem & item);
void half(uItem & item);


// List.cpp
#include "List.h"

int List::add(uItem & item)
{
    if (top < MAX)
    {
        items[top++] = item;
        return true;
    }
    else
        return false;
}

bool List::isempty()
{
    return top == 0;
}

bool List::isfull()
{
    return top == (MAX - 1);
}

void List::visit(void(*pf)(uItem &))
{
    for (int i = 0; i < top; i++)
    {
        cout << "#" << i << ": ";
        pf(items[i]);
    }
}

void oringin(uItem & item)
{
    cout << item << endl;
}

void x2(uItem & item)
{
    item = item * 2;
    cout << item << endl;
}

void half(uItem & item)
{
    item = item / 2;
    cout << item << endl;
}


// Sales.h
#pragma once
#include <iostream>

namespace SALES
{
    const int QUARTERS = 4;
    class Sales {
    private:
        double sales[QUARTERS];
        double average;
        double max;
        double min;

    public:
        Sales(double ar[], int n);
        Sales();
        void showSales();
    };
}


// Sales.cpp
#include "Sales.h"

using namespace std;
using namespace SALES;
Sales::Sales()
{
    double total = 0.0;
    double max = 0.0;
    double min = 0.0;
    double input = 0.0;

    cout << "Enter 4 double number; " << endl;
    for (size_t i = 0; i < QUARTERS; i++)
    {
        cin >> input;
        if (i == 0)
        {
            max = input;
            min = input;
        }
        if (min > input)
        {
            min = input;
        }
        if (max < input)
        {
            max = input;
        }
        total += input;
        sales[i] = input;
    }
    this->average = total / QUARTERS;
    this->max = max;
    this->min = min;
}
Sales::Sales(double ar[], int n)
{
    double total = 0.0;
    double max = ar[0];
    double min = ar[0];

    for (int i = 0; i < n; i++)
    {
        total += ar[i];
        sales[i] = ar[i];
        if (max < ar[i])
        {
            max = ar[i];
        }
        if (min > ar[i])
        {
            min = ar[i];
        }
    }
    for (int i = n; i < QUARTERS; i++)
    {
        sales[i] = 0.0;
    }

    this->average = total / n;
    this->max = max;
    this->min = min;
}
void Sales::showSales()
{
    cout << "sales: ";
    for (size_t i = 0; i < QUARTERS; i++)
    {
        cout << sales[i] << " ";
    }

    cout << endl;
    cout << "average: " << average << endl;
    cout << "max: " << max << endl;
    cout << "min: " << min << endl;
}


//stack.h
#pragma once
#ifndef STACK_H_
#define STACK_H_

struct customer
{
    char fullname[35];
    double payment;
};

typedef customer Item;

class Stack
{
private:
    enum { MAX = 10};
    Item items[MAX];
    int top;

public:
    Stack();
    bool isempty() const;
    bool isfull() const;
    bool push(const Item & item);
    bool pop(Item & item);
};

#endif // !STACK_H_


// stack.cpp
#include "stack.h"

Stack::Stack()
{
    top = 0;
}

bool Stack::isempty() const
{
    return top == 0;
}

bool Stack::isfull() const
{
    return top == MAX;
}

bool Stack::push(const Item & item)
{
    if (top < MAX)
    {
        items[top++] = item;
        return true;
    }
    else
        return false;
}

bool Stack::pop(Item & item)
{
    if (top > 0)
    {
        item = items[--top];
        return true;
    }
    else
        return false;
}

```

# 第十一章课后习题

```
// Chapter11.cpp
#include <iostream>
#include <cstring>
#include <string>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "vector.h"
#include "mytime1.h"
#include "Stonewt1.h"
#include "Stonewt2.h"
#include "complex0.h"

using namespace std;


// practice 1
// 第六版的校对做得真的有点差，随机漫步应该是程序清单11.15才对
// 对应的vector.h和vector.cpp可以用书本中原有的，也可以用practice 2中修改过的
void p11_1(void)
{
    using VECTOR::Vector;
    string filename = "randwalk.txt";
    ofstream outFile;
    outFile.open(filename.c_str(), ios_base::out);
    srand(time(0));
    double direction;
    Vector step;
    Vector result(0.0, 0.0);
    unsigned long steps = 0;
    double target;
    double dstep;
    cout << "Enter target distance (q to quit): ";
    while (cin >> target)
    {
        cout << "Enter step length: ";
        if (!(cin >> dstep))
            break;

        outFile << "Target Distance: " << target << ", " << "Step Size: " << dstep << endl;
        while (result.magval() < target)
        {
            direction = rand() % 360;
            step.reset(dstep, direction, Vector::POL);
            result = result + step;
            steps++;
            step.rect_mode();
            outFile << result << endl;
        }
        outFile << "After " << steps << " steps, the subject "
            "has the following locations:\n";
        outFile << result << endl;
        result.polar_mode();
        outFile << " or\n" << result << endl;
        outFile << "Average outward distance per step = "
            << result.magval() / steps << endl;
        steps = 0;
        result.reset(0.0, 0.0);
        cout << "Enter target distance (q to quit): ";
    }
    cout << "Bye!\n";
    cin.clear();
    while (cin.get() != '\n')
        continue;

    return ;
}


// practice 2
void p11_2(void)
{
    using VECTOR::Vector;
    srand(time(0));
    double direction;
    Vector step;
    Vector result(0.0, 0.0);
    unsigned long steps = 0;
    double target;
    double dstep;
    cout << "Enter target distance (q to quit): ";
    while (cin >> target)
    {
        cout << "Enter step length: ";
        if (!(cin >> dstep))
            break;

        cout << "Target Distance: " << target << ", " << "Step Size: " << dstep << endl;
        while (result.magval() < target)
        {
            direction = rand() % 360;
            step.reset(dstep, direction, Vector::POL);
            result = result + step;
            steps++;
            step.rect_mode();
            cout << result << endl;
        }
        cout << "After " << steps << " steps, the subject "
            "has the following locations:\n";
        cout << result << endl;
        result.polar_mode();
        cout << " or\n" << result << endl;
        cout << "Average outward distance per step = "
            << result.magval() / steps << endl;
        steps = 0;
        result.reset(0.0, 0.0);
        cout << "Enter target distance (q to quit): ";
    }
    cout << "Bye!\n";
    cin.clear();
    while (cin.get() != '\n')
        continue;

    return;
}


// practice 3
void p11_3(void)
{
    using VECTOR::Vector;
    srand(time(0));
    double direction;
    Vector step;
    Vector result(0.0, 0.0);
    unsigned long steps = 0;
    double target;
    double dstep;
    unsigned long max_steps = 0;
    unsigned long min_steps = 0 - 1;
    unsigned long total_steps = 0;
    double average_steps = 0.0;
    int test_times = 0;
    cout << "Enter target distance (q to quit): ";
    cin >> target;
    cout << "Enter step length: ";
    cin >> dstep;
    cout << "Enter test times: ";
    cin >> test_times;
    cin.get();
    for (int i = 0; i < test_times; i++)
    {
        cout << "Target Distance: " << target << ", " << "Step Size: " << dstep << endl;
        while (result.magval() < target)
        {
            direction = rand() % 360;
            step.reset(dstep, direction, Vector::POL);
            result = result + step;
            steps++;
            step.rect_mode();
            cout << result << endl;
        }
        cout << "After " << steps << " steps, the subject "
            "has the following locations:\n";
        cout << result << endl;
        result.polar_mode();
        cout << " or\n" << result << endl;
        cout << "Average outward distance per step = "
            << result.magval() / steps << endl;

        total_steps += steps;
        if (steps > max_steps)
        {
            max_steps = steps;
        }
        else if (steps < min_steps)
        {
            min_steps = steps;
        }
        steps = 0;
        result.reset(0.0, 0.0);
        cout << endl;
    }

    average_steps = (double)total_steps / test_times;
    cout << endl;
    cout << "Test " << test_times << " times, max steps is " << max_steps << ", min steps is " << min_steps << ", average steps is " << average_steps << endl;

    cout << "Bye!\n";
    cin.clear();
    while (cin.get() != '\n')
        continue;

    return;
}


// practice 4
void p11_4(void)
{
    Time aida(3, 35);
    Time tosca(2, 48);
    Time temp;

    cout << "Aida and Tosca:\n";
    cout << aida << "; " << tosca << endl;
    temp = aida + tosca;
    cout << "Aida + Tosca: " << temp << endl;

    temp = aida * 1.17;
    cout << "Aida * 1.17: " << temp << endl;

    cout << "10 * Tosca: " << 10 * tosca << endl;

    return;
}


// practice 5
void p11_5(void)
{
    Stonewt1 st1(120);
    Stonewt1 st2(12.3, Stonewt1::IPOUND);
    Stonewt1 st3(123.3, Stonewt1::DPOUND);

    Stonewt1 st4 = 13 + st1;
    Stonewt1 st5 = 13 * st2;
    Stonewt1 st6 = 13 - st2;

    Stonewt1 st7 = st1 + 10;
    Stonewt1 st8 = st1 - 10;
    Stonewt1 st9 = st1 * 10;

    cout << st1 << endl;
    cout << st2 << endl;
    cout << st3 << endl;
    cout << st4 << endl;
    cout << st5 << endl;
    cout << st6 << endl;
    cout << st7 << endl;
    cout << st8 << endl;
    cout << st9 << endl;
}


// practice 6
void p11_6(void)
{
    Stonewt2 st[6] = {
        Stonewt2(10.0, 0),
        Stonewt2(11.1, 0),
        Stonewt2(22.2, 0)
    };
    Stonewt2 max = st[0];
    Stonewt2 min = st[0];
    Stonewt2 ele(11, 0);
    int num = 0;

    for (int i = 0; i < 3; i++)
    {
        double pounds = 0;
        cout << "Enter the pounds: ";
        cin >> pounds;
        Stonewt2 tmp(pounds);
        st[3 + i] = tmp;
    }

    for (int i = 0; i < 6; i++)
    {
        if (max < st[i])
        {
            max = st[i];
        }
        else if (min > st[i])
        {
            min = st[i];
        }

        if (st[i] >= ele)
        {
            num++;
        }
    }

    cout << "Max element: ";
    max.show_stn();

    cout << "Min element: ";
    min.show_stn();

    cout << "There are " << num << " element large than 11 stones." << endl;

    return;
}


// practice 7
char *mystrncpy(char *s1, char *s2, int n)
{
    int s1_length = strlen(s1);
    int s2_length = strlen(s2);
    int copy_length = (s2_length >= n ? n : s2_length);

    for (size_t i = 0; i < copy_length; i++)
    {
        *++s1 = *++s2;
    }

    s1[copy_length - 1] = '\0';

    return s1;
}
void p11_7(void)
{
    char string1[256];
    char string2[256];
    int string1_length = 0;
    int string2_length = 0;

    while (1) // 一直循环输入吧，也可以添加一个退出条件
    {
        printf("Enter the first string:");
        fgets(string1, 256, stdin);

        printf("Enter the second string:");
        fgets(string2, 256, stdin);

        string1_length = strlen(string1);
        string2_length = strlen(string2);

        printf("The first original string: ");
        printf("%s", string1);
        mystrncpy(string1, string2, 256);
        printf("After mystrncpy, ths first string: ");
        printf("%s", string2);

        printf("\n");
    }

    return;
}


int main(int argc, char ** argv)
{
    p11_7();

    while (cin.get());
}

// complex0.h
#pragma once
#ifndef COMPLEX0_H_
#define COMPLEX0_H_
#include <iostream>

class complex
{
public:
    complex();
    complex(double real, double imaginary);
    complex operator+(const complex & cx) const;
    complex operator-(const complex & cx) const;
    complex operator*(const complex & cx) const;

    friend complex operator~(const complex & cx);
    friend complex operator*(int x, const complex & cx);
    friend std::istream & operator>>(std::istream & is, complex & cx);
    friend std::ostream & operator<<(std::ostream & os, const complex & cx);

private:
    double real;
    double imaginary;

};

#endif // !COMPLEX0_H_

// complex0.cpp
#include <iostream>
using namespace std;

#include "complex0.h"

complex::complex()
{
    real = imaginary = 0;
}

complex::complex(double real, double imaginary)
{
    this->real = real;
    this->imaginary = imaginary;
}

complex complex::operator+(const complex & cx) const
{
    return complex(real + cx.real, imaginary + cx.imaginary);
}

complex complex::operator-(const complex & cx) const
{
    return complex(real - cx.real, imaginary - cx.imaginary);
}

complex complex::operator*(const complex & cx) const
{
    return complex(real * cx.real - imaginary * cx.imaginary,
        real * cx.imaginary + imaginary * cx.real);
}

complex operator~(const complex & cx)
{
    return complex(cx.real, -cx.imaginary);
}

complex operator*(int x, const complex & cx)
{
    return complex(x*cx.real, x*cx.imaginary);
}

std::istream & operator>>(std::istream & is, complex & cx)
{
    cout << "real: ";
    is >> cx.real;
    if (!is)
    {
        return is;
    }
    cout << "imaginary: ";
    is >> cx.imaginary;
    cin.get();

    return is;
}

std::ostream & operator<<(std::ostream & os, const complex & cx)
{
    os << "(" << cx.real << ", " << cx.imaginary << "i)";

    return os;
}


// mytime1.h
#pragma once
#ifndef MYTIME1_H_
#define MYTIME1_H_

#include <iostream>

class Time
{
private:
    int hours;
    int minutes;

public:
    Time();
    Time(int h, int m = 0);
    void AddMin(int m);
    void AddHr(int h);
    void Reset(int h = 0, int m = 0);
    friend Time operator+(const Time & t1, const Time & t2);
    friend Time operator-(const Time & t1, const Time & t2);
    friend Time operator*(double n, const Time & t);
    friend Time operator*(const Time & t, double n);
    friend std::ostream & operator<<(std::ostream & os, const Time & t);

};


#endif // MYTIME1_H_


// mytime1.cpp
#include "mytime1.h"

using namespace std;

Time::Time()
{

}

Time::Time(int h, int m)
{
    hours = h;
    minutes = m;
}

void Time::AddHr(int h)
{
    hours += h;
}

void Time::AddMin(int m)
{
    minutes += m;
}

void Time::Reset(int h, int m)
{
    hours = h;
    minutes = m;
}

Time operator+(const Time & t1, const Time & t2)
{
    Time sum;
    sum.minutes = t1.minutes + t2.minutes;
    sum.hours = t1.hours + t2.hours + sum.minutes / 60;
    sum.minutes %= 60;
    return sum;
}

Time operator-(const Time & t1, const Time & t2)
{
    Time diff;
    int tot1, tot2;
    tot1 = t1.minutes + 60 * t1.hours;
    tot2 = t2.minutes + 60 * t2.hours;
    diff.minutes = (tot1 - tot2) % 60;
    diff.hours = (tot1 - tot2) / 60;
    return diff;
}

Time operator*(double n, const Time & t)
{
    Time result;
    long totalminutes = t.hours * n * 60 + t.minutes * n;
    result.hours = totalminutes / 60;
    result.minutes = totalminutes % 60;
    return result;
}

Time operator*(const Time & t, double n)
{
    Time result;
    long totalminutes = t.hours * n * 60 + t.minutes * n;
    result.hours = totalminutes / 60;
    result.minutes = totalminutes % 60;
    return result;
}

std::ostream & operator<<(std::ostream & os, const Time & t)
{
    os << t.hours << " hours, " << t.minutes << " minutes";
    return os;
}


// Stonewt1.h
#pragma once
#ifndef STONEWT1_H_
#define STONEWT1_H_

#include <iostream>

class Stonewt1
{
public:
    enum format { STONES, IPOUND, DPOUND };

private:
    enum {Lbs_per_stn = 14};
    format ft;
    double stone;
    int ipounds;
    double dpounds;

    // 有三种格式的数据，成员函数先对total进行操作
    // 然后调用update更新所有的成员变量
    // 免得每个成员函数里面都用if...else if的判断来更新
    double total;
    void update();

public:
    Stonewt1();
    Stonewt1(double n, format f = STONES);
    Stonewt1 operator+(double n) const;
    Stonewt1 operator-(double n) const;
    Stonewt1 operator*(double n) const;

    friend Stonewt1 operator+(double n, const Stonewt1 & st);
    friend Stonewt1 operator-(double n, const Stonewt1 & st);
    friend Stonewt1 operator*(double n, const Stonewt1 & st);
    friend std::ostream & operator<<(std::ostream & os, const Stonewt1 & st);
};

#endif // !STONEWT1_H_


// Stonewt1.cpp
#include "Stonewt1.h"
using namespace std;

void Stonewt1::update()
{
    if (ft == STONES)
    {
        stone = total;
        ipounds = (int)stone * Lbs_per_stn;
        dpounds = stone * Lbs_per_stn;
    }
    else if (ft == IPOUND)
    {
        ipounds = total;
        stone = ipounds / Lbs_per_stn;
        dpounds = ipounds;
    }
    else if (ft == DPOUND)
    {
        dpounds = total;
        stone = dpounds / Lbs_per_stn;
        ipounds = (int)dpounds;
    }
}

Stonewt1::Stonewt1()
{
    total = 0;
    ft = STONES;
    update();
}

Stonewt1::Stonewt1(double n, format f)
{
    total = n;
    ft = f;
    update();
}

Stonewt1 Stonewt1::operator+(double n) const
{
    Stonewt1 st(total + n, ft);
    return st;
}

Stonewt1 Stonewt1::operator-(double n) const
{
    Stonewt1 st(total - n, ft);
    return st;
}

Stonewt1 Stonewt1::operator*(double n) const
{
    Stonewt1 st(total * n, ft);
    return st;
}

Stonewt1 operator+(double n, const Stonewt1 & st)
{
    return (st + n);
}

Stonewt1 operator-(double n, const Stonewt1 & st)
{
    return st - n;
}

Stonewt1 operator*(double n, const Stonewt1 & st)
{
    return st * n;
}

std::ostream & operator<<(std::ostream & os, const Stonewt1 & st)
{
    if (st.ft == st.STONES)
    {
        cout << "Stone: " << st.stone ;
    }
    else if (st.ft == st.IPOUND)
    {
        cout << "pounds in int: " << st.ipounds ;
    }
    else if (st.ft == st.DPOUND)
    {
        cout << "pounds in double: " << st.dpounds;
    }

    return os;
}


// Stonewt2.h
#pragma once
#ifndef MYTIME2_H_
#define MYTIME2_H_

class Stonewt2
{
private:
    enum { Lbs_per_stn = 14 };
    int stone;
    double pds_left;
    double pounds;

public:
    Stonewt2(double lbs);
    Stonewt2(int stn, double lbs);
    Stonewt2();
    ~Stonewt2();
    void show_lbs() const;
    void show_stn() const;

    bool operator==(const Stonewt2 & st) const;
    bool operator!=(const Stonewt2 & st) const;
    bool operator<(const Stonewt2 & st) const;
    bool operator>(const Stonewt2 & st) const;
    bool operator<=(const Stonewt2 & st) const;
    bool operator>=(const Stonewt2 & st) const;
};

#endif // !MYTIME2_H_


// Stone2.cpp
#include "Stonewt2.h"
#include <iostream>

using namespace std;

Stonewt2::Stonewt2(double lbs)
{
    stone = int(lbs) / Lbs_per_stn;
    pds_left = int(lbs) % Lbs_per_stn + lbs - (int)lbs;
    pounds = lbs;
}

Stonewt2::Stonewt2(int stn, double lbs)
{
    stone = stn;
    pds_left = lbs;
    pounds = stn * Lbs_per_stn + lbs;
}

Stonewt2::Stonewt2()
{
    stone = pds_left = pounds = 0;
}

Stonewt2::~Stonewt2()
{

}

void Stonewt2::show_stn() const
{
    cout << stone << " stone, " << pds_left << " pounds\n";
}

void Stonewt2::show_lbs() const
{
    cout << pounds << " pounds\n";
}

bool Stonewt2::operator<(const Stonewt2 & st) const
{
    return pounds < st.pounds;
}

bool Stonewt2::operator>(const Stonewt2 & st) const
{
    return pounds > st.pounds;
}

bool Stonewt2::operator<=(const Stonewt2 & st) const
{
    return pounds <= st.pounds;
}

bool Stonewt2::operator>=(const Stonewt2 & st) const
{
    return pounds >= st.pounds;
}

bool Stonewt2::operator==(const Stonewt2 & st) const
{
    return pounds == st.pounds;
}

bool Stonewt2::operator!=(const Stonewt2 & st) const
{
    return pounds != st.pounds;
}


// vector.h
#ifndef VECTOR_H_
#define VECTOR_H_
#pragma once

#include <iostream>

namespace VECTOR
{
    class Vector
    {
    public:
        enum Mode {RECT, POL};

    private:
        double x;
        double y;
        Mode mode;
        void set_x(double, double);
        void set_y(double, double);

    public:
        Vector();
        Vector(double n1, double n2, Mode form = RECT);
        void reset(double n1, double n2, Mode form = RECT);
        ~Vector();
        double xval() const { return x; }
        double yval() const { return y; }
        double magval() const;
        double angval() const;
        void polar_mode();
        void rect_mode();

        Vector operator+(const Vector & b) const;
        Vector operator-(const Vector & b) const;
        Vector operator-() const;
        Vector operator*(double n) const;

        friend Vector operator*(double n, const Vector & a);
        friend std::ostream & operator<<(std::ostream & os, const Vector & v);
    };
}

#endif // !VECTOR_H_


// vector.cpp
#include <cmath>
#include "vector.h"

using std::sqrt;
using std::sin;
using std::cos;
using std::atan;
using std::atan2;
using std::cout;

namespace VECTOR
{
    const double Rad_to_deg = 45.0 / atan(1.0);

    void Vector::set_x(double mag, double ang)
    {
        x = mag * cos(ang);
    }

    void Vector::set_y(double mag, double ang)
    {
        y = mag * sin(ang);
    }

    Vector::Vector()
    {
        x = y = 0.0;
        mode = RECT;
    }

    Vector::Vector(double n1, double n2, Mode form)
    {
        mode = form;
        if (form == RECT)
        {
            x = n1;
            y = n2;
        }
        else if (form == POL)
        {
            set_x(n1, n2);
            set_y(n1, n2);
        }
        else
        {
            cout << "Incorrect 3rd argument to Vector() -- ";
            cout << "vector set to 0\n";
            x = y = 0.0;
            mode = RECT;
        }
    }

    void Vector::reset(double n1, double n2, Mode form)
    {
        mode = form;
        if (form == RECT)
        {
            x = n1;
            y = n2;
        }
        else if (form == POL)
        {
            set_x(n1, n2);
            set_y(n1, n2);
        }
        else
        {
            cout << "Incorrect 3rd argument to Vector() -- ";
            cout << "vector set to 0\n";
            x = y = 0.0;
            mode = RECT;
        }
    }

    Vector::~Vector()
    {

    }

    void Vector::polar_mode()
    {
        mode = POL;
    }

    void Vector::rect_mode()
    {
        mode = RECT;
    }

    double Vector::magval() const
    {
        return sqrt(x * x + y * y);
    }

    double Vector::angval() const
    {
        if(x == 0.0 && y == 0.0)
        {
            return 0.0;
        }
        else
            return atan2(y, x);
    }

    Vector Vector::operator+(const Vector & b) const
    {
        return Vector(x + b.x, y + b.y);
    }

    Vector Vector::operator-(const Vector & b) const
    {
        return Vector(x - b.x, y - b.y);
    }

    Vector Vector::operator-() const
    {
        return Vector(-x, -y);
    }

    Vector Vector::operator*(double n) const
    {
        return Vector(n*x, n*y);
    }

    Vector operator*(double n, const Vector & a)
    {
        return a * n;
    }

    std::ostream & operator<<(std::ostream & os, const Vector & v)
    {
        if (v.mode == Vector::RECT)
        {
            os << "(x,y) = (" << v.x << ", " << v.y << ")";
        }
        else if (v.mode == Vector::POL)
        {
            os << "(m,a) = (" << v.magval() << ", " << v.angval() * Rad_to_deg << ")";
        }
        else
            os << "Vector object mode is invalid";

        return os;
    }
}
```

# 第十二章课后习题

```
// Chapter12.h
#pragma once
#ifndef CHAPTER12_H_
#define CHAPTER12_H_

class Cow {
    char name[20];
    char * hobby;
    double weight;

public:
    Cow();
    Cow(const char * nm, const char * ho, double wt);
    Cow(const Cow & c);
    ~Cow();
    Cow & operator=(const Cow & c);
    void ShowCow() const;
};

#endif // !CHAPTER12_H_


// Chapter12.cpp
#include <iostream>
#include <string>
#include <cstring>
#include <cctype>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include "Chapter12.h"
#include "string2.h"
#include "Stock.h"
#include "Stack.h"
#include "Queue.h"

using namespace std;


// practice 1
Cow::Cow()
{

}
Cow::Cow(const char * nm, const char * ho, double wt)
{
    strncpy(name, nm, 20);
    int ho_length = strlen(ho);
    hobby = new char[ho_length + 1];
    strncpy(hobby, ho, ho_length);
    hobby[ho_length] = '\0';
    weight = wt;
}
Cow::Cow(const Cow & c)
{
    strncpy(name, c.name, 20);
    int ho_length = strlen(c.hobby);
    hobby = new char[ho_length + 1];
    strncpy(hobby, c.hobby, ho_length);
    hobby[ho_length] = '\0';
    weight = c.weight;
}
Cow::~Cow()
{
    delete[] hobby;
    hobby = nullptr;
}
Cow & Cow::operator=(const Cow & c)
{
    strncpy(name, c.name, 20);
    int ho_length = strlen(c.hobby);
    hobby = new char[ho_length + 1];
    strncpy(hobby, c.hobby, ho_length);
    hobby[ho_length] = '\0';
    weight = c.weight;

    return *this;
}
void Cow::ShowCow() const
{
    cout << "Cow name: " << name << endl;
    cout << "Cow hobby: " << hobby << endl;
    cout << "Cow weight: " << weight << endl;
}
void p12_1(void)
{
    Cow cow1("hehehe", "eat", 123.4);
    Cow cow2("heihei", "drink", 321.2);
    Cow cow3(cow2);
    Cow cow4 = cow1;

    cow1.ShowCow();
    cow2.ShowCow();
    cow3.ShowCow();
    cow4.ShowCow();
}


// practice 2
void p12_2(void)
{
    String2 s1("and i am a C++ student.");
    String2 s2 = "Please enter your name: ";
    String2 s3;
    cout << s2;
    cin >> s3;
    s2 = "My name is " + s3;
    cout << s2 << ".\n";
    s2 = s2 + s1;
    s2.Stringup();
    cout << "The sting\n" << s2 << "\ncontains " << s2.has('A')
        << " 'A' characters in it.\n";
    s1 = "red";
    String2 rgb[3] = { String2(s1), String2("green"), String2("blue") };
    cout << "Enter the name of a primary color for mixing light: ";
    String2 ans;
    bool success = false;
    while (cin >> ans)
    {
        ans.Stringlow();
        for (int i = 0; i < 3; i++)
        {
            if (ans == rgb[i])
            {
                cout << "That's right!\n";
                success = true;
                break;
            }
        }
        if (success)
        {
            break;
        }
        else
        {
            cout << "Try again!\n";
        }
    }

    cout << "Bye!\n";
    return ;
}


// practice 3
const int STKS = 4;
void p12_3(void)
{
    Stock stocks[STKS] = {
        Stock("NanoSmart", 12, 20.0),
        Stock("Boffo Objects", 200, 2.0),
        Stock("Monolithic Obelisks", 130, 3.25),
        Stock("Fleep Enterprises", 60, 6.5)
    };

    std::cout << "Stock holdings:\n";
    int st;
    for (st = 0; st < STKS; st++)
    {
        std::cout << stocks[st];
    }

    const Stock * top = &stocks[0];
    for (st = 0; st < STKS; st++)
    {
        top = &top->topval(stocks[st]);
    }

    std::cout << "\nMost valuable holding:\n";
    cout << *top;
    return;
}


// practice 4
void p12_4(void)
{
    Stack st1(10);
    srand(time(0));
    for (size_t i = 0; i < 10; i++)
    {
        if (!st1.push(rand() % 100))
        {
            cout << "Push error!" << endl;;
        }

    }
    if (!st1.push(0))
    {
        cout << "Push 0 error!" << endl;
    }
    Stack st2(st1);
    Stack st3 = st1;
    // 故意导致pop error
    for (size_t i = 0; i < 11; i++)
    {
        Item item;
        cout << "#" << i+1 << ": " << endl;
        if (!st1.pop(item))
        {
            cout << "st1 pop error!" << endl;
        }
        else
            cout << "st1: " << item << endl;

        if (!st2.pop(item))
        {
            cout << "st2 pop error!" << endl;
        }
        else
            cout << "st2: " << item << endl;

        if (!st3.pop(item))
        {
            cout << "st3 pop error!" << endl;
        }
        else
            cout << "st3: " << item << endl;

        cout << endl;
    }
}


// practice 5
const int MIN_PER_HR = 60;
bool newcustomer(double x)
{
    return (std::rand() * x / RAND_MAX < 1);
}
void p12_5(void)
{
    using std::cin;
    using std::cout;
    using std::endl;
    using std::ios_base;

    std::srand(std::time(0));

    cout << "Case Study: Bank of Heather Automatic Teller\n";
    cout << "Enter maximun size of queue: ";
    int qs;
    cin >> qs;
    Queue line(qs);

    // 直接将测试时间设置为100好了
    cout << "Simulation hours: 100" << endl;
    int hours = 100;

    long cyclelimit = MIN_PER_HR * hours;

    cItem temp;
    long turnaways = 0;
    long customers = 0;
    long served = 0;
    long sum_line = 0;
    int wait_time = 0;
    long line_wait = 0;
    double avg_wait_time = 0.0;

    // 每小时使用人数初始值为15，之后进行+1枚举测试
    double perhour = 15;
    double min_per_cust;

    do {
        min_per_cust = MIN_PER_HR / perhour;
        turnaways = 0;
        customers = 0;
        served = 0;
        sum_line = 0;
        wait_time = 0;
        line_wait = 0;
        avg_wait_time = 0;

        // 需要清空队列
        while (!line.isempty())
        {
            line.dequeue(temp);
        }

        for (int cycle = 0; cycle < cyclelimit; cycle++)
        {
            if (newcustomer(min_per_cust))
            {
                if (line.isfull())
                {
                    turnaways++;
                }
                else
                {
                    customers++;
                    temp.set(cycle);
                    line.enqueue(temp);
                }
            }
            if (wait_time <= 0 && !line.isempty())
            {
                line.dequeue(temp);
                wait_time = temp.ptime();
                line_wait += (cycle - temp.when());
                served++;
            }
            if (wait_time > 0)
            {
                wait_time--;
            }
            sum_line += line.queuecount();
        }

        if (customers > 0)
        {
            cout << "customers accepted: " << customers << endl;
            cout << "  customers served: " << served << endl;
            cout << "         turnaways: " << turnaways << endl;
            cout << "average queue size: ";
            cout.precision(2);
            cout.setf(ios_base::fixed, ios_base::floatfield);
            cout << (double)sum_line / cyclelimit << endl;
            avg_wait_time = (double)line_wait / served;
            cout << " average wait time: "
                << avg_wait_time << " minutes\n" << endl;
        }
        else
            cout << "No customers!\n" << endl;;
        // 平均等待时间不一定刚好是1，设置在一个区间内
    } while ((perhour++) && (avg_wait_time < 0.9) || (avg_wait_time > 1.1));

    cout << "When perhour = " << perhour << " , the average wait time"
        " will be about 1 minute\n" << endl;

    cout << "Done!\n";

    return;
}


// practice 6
void p12_6(void)
{
    using std::cin;
    using std::cout;
    using std::endl;
    using std::ios_base;

    std::srand(std::time(0));

    cout << "Case Study: Bank of Heather Automatic Teller\n";
    cout << "Enter maximun size of queue: ";
    int qs;
    cin >> qs;
    Queue line1(qs);
    Queue line2(qs);

    // 直接将测试时间设置为100好了
    cout << "Simulation hours: 100" << endl;
    int hours = 100;

    long cyclelimit = MIN_PER_HR * hours;

    cItem temp;
    long turnaways = 0;
    long customers = 0;
    long served = 0;
    long sum_line = 0;
    int wait_time1 = 0;
    int wait_time2 = 0;
    int line1_size = 0;
    int line2_size = 0;
    long line_wait = 0;
    double avg_wait_time = 0.0;

    // 每小时使用人数初始值为15，之后进行+1枚举测试
    double perhour = 15;
    double min_per_cust;

    do {
        min_per_cust = MIN_PER_HR / perhour;
        turnaways = 0;
        customers = 0;
        served = 0;
        sum_line = 0;
        wait_time1 = 0;
        wait_time2 = 0;
        line1_size = 0;
        line2_size = 0;
        line_wait = 0;
        avg_wait_time = 0;

        // 需要清空队列
        while (!line1.isempty())
        {
            line1.dequeue(temp);
        }

        while (!line2.isempty())
        {
            line2.dequeue(temp);
        }

        for (int cycle = 0; cycle < cyclelimit; cycle++)
        {
            if (newcustomer(min_per_cust))
            {
                if (line1.isfull() && line2.isfull())
                {
                    turnaways++;
                }
                else if (line1_size > line2_size)
                {
                    customers++;
                    temp.set(cycle);
                    line2.enqueue(temp);
                    line2_size++;
                } 
                else
                {
                    customers++;
                    temp.set(cycle);
                    line1.enqueue(temp);
                    line1_size++;
                }
            }
            if (wait_time1 <= 0 && !line1.isempty())
            {
                line1.dequeue(temp);
                line1_size--;
                wait_time1 = temp.ptime();
                line_wait += (cycle - temp.when());
                served++;
            }
            if (wait_time2 <=0 && !line2.isempty())
            {
                line2.dequeue(temp);
                line2_size--;
                wait_time2 = temp.ptime();
                line_wait += (cycle - temp.when());
                served++;
            }
            if (wait_time1 > 0)
            {
                wait_time1--;
            }
            if (wait_time2 > 0)
            {
                wait_time2--;
            }
            sum_line += line1.queuecount();
            sum_line += line2.queuecount();
        }

        if (customers > 0)
        {
            cout << "customers accepted: " << customers << endl;
            cout << "  customers served: " << served << endl;
            cout << "         turnaways: " << turnaways << endl;
            cout << "average queue size: ";
            cout.precision(2);
            cout.setf(ios_base::fixed, ios_base::floatfield);
            cout << (double)sum_line / cyclelimit << endl;
            avg_wait_time = (double)line_wait / served;
            cout << " average wait time: "
                << avg_wait_time << " minutes\n" << endl;;
        }
        else
            cout << "No customers!\n" << endl;
        // 平均等待时间不一定刚好是1，设置在一个区间内
    } while ((perhour++) && (avg_wait_time < 0.9) || (avg_wait_time > 1.1));

    cout << "When perhour = " << perhour << " , the average wait time"
        " will be about 1 minute\n" << endl;

    cout << "Done!\n";

    return;
}


int main(int argc, char **argv)
{
    p12_6();

    while (cin.get());
}


// Queue.h
#pragma once

class Customer
{
private:
    long arrive;
    int processtime;

public:
    Customer() { arrive = processtime = 0; }
    void set(long when);
    long when() const { return arrive; }
    int ptime() const { return processtime; }
};

typedef Customer cItem;

class Queue
{
private:
    struct Node { cItem item; struct Node * next; };
    enum {Q_SIZE = 10};
    Node * front;
    Node * rear;
    int items;
    const int qsize;
    Queue(const Queue & q) : qsize(0) {}
    Queue & operator=(const Queue & q) { return *this; }

public:
    Queue(int qs = Q_SIZE);
    ~Queue();
    bool isempty() const;
    bool isfull() const;
    int queuecount() const;
    bool enqueue(const cItem & item);
    bool dequeue(cItem & item);
};


// Queue.cpp
#include "Queue.h"
#include <cstdlib>


Queue::Queue(int qs) : qsize(qs)
{
    front = rear = NULL;
    items = 0;
}

Queue::~Queue()
{
    Node * temp;
    while (front != NULL)
    {
        temp = front;
        front = front->next;
        delete temp;
    }
}

bool Queue::isempty() const
{
    return items == 0;
}

bool Queue::isfull() const
{
    return items == qsize;
}

int Queue::queuecount() const
{
    return items;
}

bool Queue::enqueue(const cItem & item)
{
    if (isfull())
    {
        return false;
    }

    Node * add = new Node;
    add->item = item;
    add->next = NULL;
    items++;
    if (front == NULL)
    {
        front = add;
    }
    else
        rear->next = add;

    rear = add;
    return true;
}

bool Queue::dequeue(cItem & item)
{
    if (front == NULL)
    {
        return false;
    }

    item = front->item;
    items--;
    Node * temp = front;
    front = front->next;
    delete temp;

    if (items == 0)
    {
        rear = NULL;
    }
    return true;
}

void Customer::set(long when)
{
    processtime = std::rand() % 3 + 1;
    arrive = when;
}


// Stack.h
#pragma once
typedef unsigned long Item;


class Stack
{
private:
    enum {MAX = 10};
    Item * pitems;
    int size;
    int top;

public:
    Stack(int n = MAX);
    Stack(const Stack & st);
    ~Stack();
    bool isempty() const;
    bool isfull() const;
    bool push(const Item & item);
    bool pop(Item & item);
    Stack & operator=(const Stack & st);
};


// Stack.cpp
#include "Stack.h"
#include <iostream>


Stack::Stack(int n)
{
    pitems = new Item[n];
    size = n;
    top = 0;
}

Stack::Stack(const Stack & st)
{
    size = st.size;
    pitems = new Item[size];
    top = st.top;
    for (int i = 0; i < top; i++)
    {
        pitems[i] = st.pitems[i];
    }
}

Stack::~Stack()
{
    delete[] pitems;
    pitems = nullptr;
}

bool Stack::isempty() const
{
    return top == 0;
}

bool Stack::isfull() const
{
    return top == size;
}

bool Stack::push(const Item & item)
{
    if (isfull())
    {
        return false;
    }

    pitems[top++] = item;
    return true;
}

bool Stack::pop(Item & item)
{
    if (isempty())
    {
        return false;
    }

    item = pitems[--top];
    return true;
}

Stack & Stack::operator=(const Stack & st)
{
    size = st.size;
    pitems = new Item[size];
    top = st.top;
    for (int i = 0; i < top; i++)
    {
        pitems[i] = st.pitems[i];
    }

    return *this;
}


// Stock.h
#pragma once
#include <iostream>
class Stock
{
private:
    char * company;
    int shares;
    double share_val;
    double total_val;
    void set_tot() { total_val = shares * share_val; }

public:
    Stock();
    Stock(const char * co, long n = 0, double pr = 0.0);
    ~Stock();
    void buy(long num, double price);
    void sell(long num, double price);
    void update(double price);
    void show() const;
    friend std::ostream & operator<<(std::ostream & os, const Stock & st);
    const Stock & topval(const Stock & s) const;
};


// Stock.cpp
#include "Stock.h"
#include <cstring>


Stock::Stock()
{
    int len = std::strlen("no name");
    company = new char[len + 1];
    std::strcpy(company, "no name");
    company[len] = '\0';

    shares = 0;
    share_val = 0.0;
    total_val = 0.0;
}

Stock::Stock(const char * co, long n, double pr)
{
    int len = std::strlen(co);
    company = new char[len + 1];
    std::strcpy(company, co);

    if (n < 0)
    {
        std::cout << "Number of shares can't be negative; "
            << company << " shares set to n.\n";
        shares = 0;
    }
    else
        shares = n;
    share_val = pr;
    set_tot();
}

Stock::~Stock()
{
    delete[] company;
    company = nullptr;
}

void Stock::buy(long num, double price)
{
    if (num < 0)
    {
        std::cout << "Number of shares purchased can't be negative. "
            << "Transaction is aborted.\n";
    }
    else
    {
        shares += num;
        share_val = price;
        set_tot();
    }
}

void Stock::sell(long num, double price)
{
    using std::cout;
    if (num < 0)
    {
        cout << "Number of shares sold can't be negative. "
            << "Transaction is aborted.\n";
    }
    else if (num > shares)
    {
        cout << "You can't sell more than you have! "
            << "Transaction is aborted.\n";
    }
    else
    {
        shares -= num;
        share_val = price;
        set_tot();
    }
}

void Stock::update(double price)
{
    share_val = price;
    set_tot();
}

void Stock::show() const
{
    using std::cout;
    using std::ios_base;
    ios_base::fmtflags orig =
        cout.setf(ios_base::fixed, ios_base::floatfield);
    std::streamsize prec = cout.precision(3);

    cout << "Company: " << company
        << " Shares: " << shares << '\n';
    cout << "Share Price: $" << share_val;
    cout.precision(2);
    cout << " Total Worth: $" << total_val << '\n';

    cout.setf(orig, ios_base::floatfield);
    cout.precision(prec);
}

std::ostream & operator<<(std::ostream & os, const Stock & st)
{
    using std::ios_base;
    ios_base::fmtflags orig =
        os.setf(ios_base::fixed, ios_base::floatfield);
    std::streamsize prec = os.precision(3);

    os << "Company: " << st.company
        << " Shares: " << st.shares << '\n';
    os << " Share Price: $" << st.share_val;
    os.precision(2);
    os << " Total Worth: $" << st.total_val << '\n';

    os.setf(orig, ios_base::floatfield);
    os.precision(prec);

    return os;
}

const Stock & Stock::topval(const Stock & s) const
{
    if (s.total_val > total_val)
    {
        return s;
    }
    else
        return *this;
}


// string2.h
#pragma once
#ifndef STRING2_H_
#define STRING2_H_

#include <iostream>
using std::ostream;
using std::istream;

class String2
{
private:
    char * str;
    int len;
    static int num_strings;
    static const int CINLIM = 80;

public:
    String2(const char * s);
    String2();
    String2(const String2 &);
    ~String2();
    int length() const { return len; }

    String2 & operator=(const String2 &);
    String2 & operator=(const char *);
    char & operator[](int i);
    const char & operator[](int i) const;

    friend bool operator<(const String2 &st, const String2 &st2);
    friend bool operator>(const String2 &st, const String2 &st2);
    friend bool operator==(const String2 &st, const String2 &st2);
    friend ostream & operator<<(ostream & os, const String2 &st);
    friend istream & operator>>(istream & is, String2 &st);

    static int HowMany();

    String2 operator+(const String2 &) const;
    String2 operator+(const char *) const;
    friend String2 operator+(const char *, const String2 &);
    void Stringlow();
    void Stringup();
    int has(char);
};

#endif // !STRING2_H_


// string2.cpp
#include <cstring>
#include "string2.h"

using std::cin;
using std::cout;

int String2::num_strings = 0;

int String2::HowMany()
{
    return num_strings;
}

String2::String2(const char * s)
{
    len = std::strlen(s);
    str = new char[len + 1];
    std::strcpy(str, s);
    num_strings++;
}

String2::String2()
{
    len = 4;
    str = new char[1];
    str[0] = '\0';
    num_strings++;
}

String2::String2(const String2 & st)
{
    num_strings++;
    len = st.len;
    str = new char[len + 1];
    std::strcpy(str, st.str);
}

String2::~String2()
{
    --num_strings;
    delete[] str;
}

String2 & String2::operator=(const String2 & st)
{
    if (this == &st)
    {
        return *this;
    }
    delete[] str;
    len = st.len;
    str = new char[len + 1];
    std::strcpy(str, st.str);
    return *this;
}

String2 & String2::operator=(const char * s)
{
    delete[] str;
    len = std::strlen(s);
    str = new char[len + 1];
    std::strcpy(str, s);
    return *this;
}

char & String2::operator[](int i)
{
    return str[i];
}

const char & String2::operator[](int i) const
{
    return str[i];
}

bool operator<(const String2 & st, const String2 & st2)
{
    return(std::strcmp(st.str, st2.str) < 0);
}

bool operator>(const String2 & st, const String2 & st2)
{
    return(std::strcmp(st.str, st2.str) > 0);
}

bool operator==(const String2 & st, const String2 & st2)
{
    return(std::strcmp(st.str, st2.str) == 0);
}

ostream & operator<<(ostream & os, const String2 & st)
{
    os << st.str;
    return os;
}

istream & operator>>(istream & is, String2 & st)
{
    char temp[String2::CINLIM];
    is.get(temp, String2::CINLIM);
    if (is)
    {
        st = temp;
    }
    while (is && is.get() != '\n')
    {
        continue;
    }
    return is;
}

String2 String2::operator+(const String2 & st) const
{
    int total_len = len + st.len;
    char *temp = new char[total_len + 1];
    std::strcpy(temp, str);
    std::strcpy(temp + len, st.str);
    temp[total_len] = '\0';
    return String2(temp);
}

String2 String2::operator+(const char * s) const
{
    int total_len = len + strlen(s);
    char *temp = new char[total_len + 1];
    std::strcpy(temp, str);
    std::strcpy(temp + len, s);
    temp[total_len] = '\0';
    return String2(temp);
}

String2 operator+(const char * s, const String2 & st)
{
    return String2(s) + st;
}

void String2::Stringlow()
{
    for (int i = 0; i < len + 1; i++)
    {
        str[i] = tolower(str[i]);
    }
}

void String2::Stringup()
{
    for (int i = 0; i < len + 1; i++)
    {
        str[i] = toupper(str[i]);
    }
}

int String2::has(char c)
{
    int same = 0;
    for (int i = 0; i < len; i++)
    {
        if (str[i] == c)
        {
            same++;
        }
    }

    return same;
}

```

# 第十三章课后习题

```
// Chapter13.cpp
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cctype>

using namespace std;

#include "Cd.h"
#include "Cd1.h"
#include "dma.h"
#include "Port.h"


// practice 1
void Bravo(const Cd & disk);
void p13_1(void)
{
    Cd c1("Beatles", "Capitol", 14, 35.5);
    Classic c2 = Classic("Piano Sonata in B flat, Fantasia in C",
        "Alfred Brendel", "Philips", 2, 57.17);
    Cd *pcd = &c1;

    cout << "Using object directly:\n";
    c1.Report();
    c2.Report();

    cout << "Using type cd * pointer to objects:\n";
    pcd->Report();
    pcd = &c2;
    pcd->Report();

    cout << "Calling a function whit a cd reference argument:\n";
    Bravo(c1);
    Bravo(c2);

    cout << "Testing assignment: ";
    Classic copy;
    copy = c2;
    copy.Report();

    return;
}
void Bravo(const Cd & disk)
{
    disk.Report();
}


// practice 2
void Bravo(const Cd1 & disk)
{
    disk.Report();
}
void p13_2(void)
{
    Cd1 c1("Beatles", "Capitol", 14, 35.5);
    Classic1 c2 = Classic1("Piano Sonata in B flat, Fantasia in C",
        "Alfred Brendel", "Philips", 2, 57.17);
    Cd1 *pcd = &c1;

    cout << "Using object directly:\n";
    c1.Report();
    c2.Report();

    cout << "Using type cd * pointer to objects:\n";
    pcd->Report();
    pcd = &c2;
    pcd->Report();

    cout << "Calling a function whit a cd reference argument:\n";
    Bravo(c1);
    Bravo(c2);

    cout << "Testing assignment: ";
    Classic1 copy;
    copy = c2;
    copy.Report();

    return;
}


// practice 3
const int CLIMENTS = 5;
void p13_3(void)
{
    baseDMA * dma[CLIMENTS];
    string label;
    int rating;
    char choice;
    string color;
    string style;

    for (int i = 0; i < CLIMENTS; i++)
    {
        cout << "Enter lable: ";
        getline(cin, label);
        cout << "Enter rating: ";
        cin >> rating;
        cout << "Enter 1 for baseDMA or 2"
            " for lacksDMA or 3 for hasDMA: ";
        while (cin >> choice && (choice != '1' && choice != '2' && choice != '3'))
        {
            cout << "Enter 1 or 2 or 3: ";
        }
        while (cin.get() != '\n');
        if (choice == '1')
        {
            dma[i] = new baseDMA(label.c_str(), rating);
        }
        else if (choice == '2')
        {
            cout << "Enter color: ";
            getline(cin, color);
            dma[i] = new lacksDMA(color.c_str(), label.c_str(), rating);
        }
        else
        {
            cout << "Enter style: ";
            getline(cin, style);
            dma[i] = new hasDMA(style.c_str(), label.c_str(), rating);
        }

    }
    cout << endl;
    for (int i = 0; i < CLIMENTS; i++)
    {
        dma[i]->View();
        cout << endl;
    }

    for (int i = 0; i < CLIMENTS; i++)
    {
        delete dma[i];
    }

    cout << "Done.\n";
    return;

}


// practice 4
void p13_4(void)
{
    Port port1("Gallo", "tawny", 20);
    VintagePort vp("None", 20, "The Noble", 1997);

    port1.Show();
    vp.Show();

    VintagePort vp2 = vp;
    Port port2 = port1;

    cout << vp2 << endl;
    cout << port2 << endl;
}


int main(int argc, char ** argv)
{
    p13_4();

    while (cin.get());
}


// Cd.h
#pragma once
#ifndef CD_H_
#define CD_H_

#include <iostream>

class Cd
{
private:
    char performers[50];
    char label[20];
    int selections;
    double playtime;

public:
    Cd(char * s1, char * s2, int n, double x);
    Cd(const Cd & d);
    Cd();
    virtual ~Cd();
    virtual void Report() const;
    virtual Cd & operator=(const Cd & d);
};

class Classic : public Cd
{
private:
    char mainfile[20];

public:
    Classic(char * s1, char * s2, char * s3, int n, double x);
    Classic(const Cd & d, char * s3);
    Classic(const Classic & d);
    Classic();
    virtual ~Classic();
    virtual void Report() const;
    virtual Classic & operator=(const Classic & d);
};

#endif // !CD_H_

// Cd.cpp
#include "Cd.h"

#include <cstring>

Cd::Cd(char * s1, char * s2, int n, double x)
{
    std::strcpy(performers, s1);
    std::strcpy(label, s2);
    selections = n;
    playtime = x;
}

Cd::Cd(const Cd & d)
{
    std::strcpy(performers, d.performers);
    std::strcpy(label, d.label);
    selections = d.selections;
    playtime = d.playtime;
}

Cd::Cd()
{
    std::strcpy(performers, "null");
    std::strcpy(label, "null");
    selections = 0;
    playtime = 0;
}


Cd::~Cd()
{
}

void Cd::Report() const
{
    std::cout << "Performers: " << performers << std::endl;
    std::cout << "Label: " << label << std::endl;
    std::cout << "Selections: " << selections << std::endl;
    std::cout << "Playtime: " << playtime << std::endl;
}

Cd & Cd::operator=(const Cd & d)
{
    std::strcpy(performers, d.performers);
    std::strcpy(label, d.label);
    selections = d.selections;
    playtime = d.playtime;

    return *this;
}



// Classic
Classic::Classic(char * s1, char * s2, char * s3, int n, double x) :
    Cd(s1, s2, n, x)
{
    std::strcpy(mainfile, s3);
}

Classic::Classic(const Cd & d, char * s3) : Cd(d)
{
    std::strcpy(mainfile, s3);
}

Classic::Classic(const Classic & d) : Cd(d)
{
    std::strcpy(mainfile, d.mainfile);
}

Classic::Classic() : Cd()
{
    std::strcpy(mainfile, "null");
}

Classic::~Classic()
{

}

void Classic::Report() const
{
    Cd::Report();
    std::cout << "Mainfile: " << mainfile << std::endl;
}

Classic & Classic::operator=(const Classic & d)
{
    Cd::operator=(d);
    std::strcpy(mainfile, d.mainfile);
    return *this;
}


// Cd1.h
#pragma once
#ifndef CD1_H_
#define CD1_H_

#include <iostream>

class Cd1
{
private:
    char * performers;
    char * label;
    int selections;
    double playtime;

public:
    Cd1(char * s1, char * s2, int n, double x);
    Cd1(const Cd1 & d);
    Cd1();
    virtual ~Cd1();
    virtual void Report() const;
    virtual Cd1 & operator=(const Cd1 & d);
};

class Classic1 : public Cd1
{
private:
    char * mainfile;

public:
    Classic1(char * s1, char * s2, char * s3, int n, double x);
    Classic1(const Cd1 & d, char * s3);
    Classic1(const Classic1 & d);
    Classic1();
    virtual ~Classic1();
    virtual void Report() const;
    virtual Classic1 & operator=(const Classic1 & d);
};

#endif // !CD1_H_

// Cd1.cpp
#include "Cd1.h"

#include <cstring>

Cd1::Cd1(char * s1, char * s2, int n, double x)
{
    int performers_length = std::strlen(s1);
    performers = new char[performers_length + 1];

    int label_length = std::strlen(s2);
    label = new char[label_length + 1];

    std::strcpy(performers, s1);
    std::strcpy(label, s2);
    selections = n;
    playtime = x;
}

Cd1::Cd1(const Cd1 & d)
{
    int performers_length = std::strlen(d.performers);
    performers = new char[performers_length + 1];

    int label_length = std::strlen(d.label);
    label = new char[label_length + 1];

    std::strcpy(performers, d.performers);
    std::strcpy(label, d.label);
    selections = d.selections;
    playtime = d.playtime;
}

Cd1::Cd1()
{
    int null_length = std::strlen("null");
    performers = new char[null_length + 1];
    label = new char[null_length + 1];

    std::strcpy(performers, "null");
    std::strcpy(label, "null");
    selections = 0;
    playtime = 0;
}


Cd1::~Cd1()
{
    delete[] performers;
    delete[] label;
    performers = nullptr;
    label = nullptr;
}

void Cd1::Report() const
{
    std::cout << "Performers: " << performers << std::endl;
    std::cout << "Label: " << label << std::endl;
    std::cout << "Selections: " << selections << std::endl;
    std::cout << "Playtime: " << playtime << std::endl;
}

Cd1 & Cd1::operator=(const Cd1 & d)
{
    delete[] performers;
    delete[] label;
    performers = nullptr;
    label = nullptr;

    int performers_length = std::strlen(d.performers);
    performers = new char[performers_length + 1];

    int label_length = std::strlen(d.label);
    label = new char[label_length + 1];

    std::strcpy(performers, d.performers);
    std::strcpy(label, d.label);
    selections = d.selections;
    playtime = d.playtime;

    return *this;
}



// Classic1
Classic1::Classic1(char * s1, char * s2, char * s3, int n, double x) :
    Cd1(s1, s2, n, x)
{
    int mainfile_length = std::strlen(s3);
    mainfile = new char[mainfile_length + 1];

    std::strcpy(mainfile, s3);
}

Classic1::Classic1(const Cd1 & d, char * s3) : Cd1(d)
{
    int mainfile_length = std::strlen(s3);
    mainfile = new char[mainfile_length + 1];

    std::strcpy(mainfile, s3);
}

Classic1::Classic1(const Classic1 & d) : Cd1(d)
{
    int mainfile_length = std::strlen(d.mainfile);
    mainfile = new char[mainfile_length + 1];

    std::strcpy(mainfile, d.mainfile);
}

Classic1::Classic1() : Cd1()
{
    int null_length = std::strlen("null");
    mainfile = new char[null_length + 1];

    std::strcpy(mainfile, "null");
}

Classic1::~Classic1()
{
    delete[] mainfile;
    mainfile = nullptr;
}

void Classic1::Report() const
{
    Cd1::Report();
    std::cout << "Mainfile: " << mainfile << std::endl;
}

Classic1 & Classic1::operator=(const Classic1 & d)
{
    Cd1::operator=(d);
    delete[] mainfile;
    mainfile = nullptr;

    int mainfile_length = std::strlen(d.mainfile);
    mainfile = new char[mainfile_length + 1];

    std::strcpy(mainfile, d.mainfile);
    return *this;
}


// dma.h
#pragma once
#ifndef DMA_H_
#define DMA_H_

#include <iostream>

class DMA
{
public:
    DMA();
    ~DMA();
    virtual void View() = 0;
};


class baseDMA : public DMA
{
private:
    char * label;
    int rating;

public:
    baseDMA(const char * l = "null", int r = 0);
    baseDMA(const baseDMA & rs);
    virtual ~baseDMA();
    baseDMA & operator=(const baseDMA & rs);
    friend std::ostream & operator<<(std::ostream & os, const baseDMA & rs);

    virtual void View();
};


class lacksDMA : public baseDMA
{
private:
    enum { COL_LEN = 40 };
    char color[COL_LEN];

public:
    lacksDMA(const char * c = "black", const char * l = "null", int r = 0);
    lacksDMA(const char * c, const baseDMA & rs);
    friend std::ostream & operator<<(std::ostream & os, const lacksDMA & rs);

    virtual void View();
};

class hasDMA : public baseDMA
{
private:
    char * style;

public:
    hasDMA(const char * s = "none", const char * l = "null", int r = 0);
    hasDMA(const char * s, const baseDMA & rs);
    hasDMA(const hasDMA & hs);
    ~hasDMA();
    hasDMA & operator=(const hasDMA & rs);
    friend std::ostream & operator<<(std::ostream & os, const hasDMA & rs);

    virtual void View();
};

#endif // !DMA_H_

// dma.cpp
#include "dma.h"

#include <cstring>


DMA::DMA()
{

}


DMA::~DMA()
{
}

baseDMA::baseDMA(const char * l, int r)
{
    int label_length = std::strlen(l);
    label = new char[label_length + 1];

    std::strcpy(label, l);
    rating = r;
}

baseDMA::baseDMA(const baseDMA & rs)
{
    int label_length = std::strlen(rs.label);
    label = new char[label_length + 1];

    std::strcpy(label, rs.label);
    rating = rs.rating;
}

baseDMA::~baseDMA()
{
    delete[] label;
    label = nullptr;
}

baseDMA & baseDMA::operator=(const baseDMA & rs)
{
    delete[] label;

    int label_length = std::strlen(rs.label);
    label = new char[label_length + 1];

    std::strcpy(label, rs.label);
    rating = rs.rating;

    return *this;
}

std::ostream & operator<<(std::ostream & os, const baseDMA & rs)
{
    os << "Label: " << rs.label << std::endl;
    os << "Rating: " << rs.rating;

    return os;
}

void baseDMA::View()
{
    std::cout << "Label: " << label << std::endl;
    std::cout << "Rating: " << rating << std::endl;
}

lacksDMA::lacksDMA(const char * c, const char * l, int t) :
    baseDMA(l, t)
{
    std::strcpy(color, c);
}

lacksDMA::lacksDMA(const char * c, const baseDMA & rs) :
    baseDMA(rs)
{
    std::strcpy(color, c);
}

std::ostream & operator<<(std::ostream & os, const lacksDMA & rs)
{
    operator<<(os, (baseDMA)rs) << std::endl;
    os << "Color: " << rs.color;

    return os;
}

void lacksDMA::View()
{
    baseDMA::View();
    std::cout << "Color: " << color << std::endl;
}

hasDMA::hasDMA(const char * s, const char * l, int r) :
    baseDMA(l, r)
{
    int style_length = std::strlen(s);
    style = new char[style_length + 1];

    std::strcpy(style, s);
}

hasDMA::hasDMA(const char * s, const baseDMA & rs) :
    baseDMA(rs)
{
    int style_length = std::strlen(s);
    style = new char[style_length + 1];

    std::strcpy(style, s);
}

hasDMA::hasDMA(const hasDMA & hs) :
    baseDMA(hs)
{
    int style_length = std::strlen(hs.style);
    style = new char[style_length + 1];

    std::strcpy(style, hs.style);
}

hasDMA::~hasDMA()
{
    delete[] style;
    style = nullptr;
}

hasDMA & hasDMA::operator=(const hasDMA & rs)
{
    baseDMA::operator=(rs);
    delete[] style;

    int style_length = std::strlen(rs.style);
    style = new char[style_length + 1];

    std::strcpy(style, rs.style);

    return *this;
}

std::ostream & operator<<(std::ostream & os, const hasDMA & rs)
{
    operator<<(os, baseDMA(rs)) << std::endl;
    os << "Style: " << rs.style;

    return os;
}

void hasDMA::View()
{
    baseDMA::View();
    std::cout << "Style: " << style << std::endl;
}


// port.h
#pragma once
#include <iostream>
#include <cstring>
using namespace std;

class Port
{
private:
    char * brand;
    char style[20];
    int bottles;

public:
    Port(const char * br = "none", const char * st = "none", int b = 0);
    Port(const Port & p);
    virtual ~Port() { delete[] brand; }
    Port & operator=(const Port & p);
    Port & operator+=(int b);
    Port & operator-=(int b);
    int BottleCount() const { return bottles; }
    virtual void Show() const;
    friend ostream & operator<<(ostream & os, const Port & p);
};

class VintagePort : public Port
{
private:
    char * nickname;
    int year;

public:
    VintagePort();
    VintagePort(const char * br, int b, const char * nn, int y);
    VintagePort(const VintagePort & vp);
    ~VintagePort() { delete[] nickname; }
    VintagePort & operator=(const VintagePort & vp);
    void Show() const;
    friend ostream & operator<<(ostream & os, const VintagePort & vp);
};

// port.cpp
#include "Port.h"



Port::Port(const char * br, const char * st, int b)
{
    int brand_length = std::strlen(br);
    brand = new char[brand_length + 1];
    std::strcpy(brand, br);
    std::strcpy(style, st);
    bottles = b;
}

Port::Port(const Port & p)
{
    int brand_length = std::strlen(p.brand);
    brand = new char[brand_length + 1];
    std::strcpy(brand, p.brand);
    std::strcpy(style, p.style);
    bottles = p.bottles;
}

Port & Port::operator=(const Port & p)
{
    delete[] brand;

    int brand_length = std::strlen(p.brand);
    brand = new char[brand_length + 1];
    std::strcpy(brand, p.brand);
    std::strcpy(style, p.style);
    bottles = p.bottles;

    return *this;
}

Port & Port::operator+=(int b)
{
    bottles += b;

    return *this;
}

Port & Port::operator-=(int b)
{
    bottles -= b;
    return *this;
}

void Port::Show() const
{
    std::cout << "Brand: " << brand << std::endl;
    std::cout << "Style: " << style << std::endl;
    std::cout << "Bottles: " << bottles << std::endl;
}

ostream & operator<<(ostream & os, const Port & s)
{
    os << s.brand << ", " << s.style << ", " << s.bottles;

    return os;
}

VintagePort::VintagePort() : Port()
{
    int null_length = std::strlen("null");
    nickname = new char[null_length + 1];
    std::strcpy(nickname, "null");
    year = 0;
}

VintagePort::VintagePort(const char * br, int b, const char * nn, int y) :
    Port(br, "Vintage", b)
{
    int nn_length = std::strlen(nn);
    nickname = new char[nn_length + 1];
    std::strcpy(nickname, nn);
    year = y;
}

VintagePort::VintagePort(const VintagePort & vp) : Port(vp)
{
    int nn_length = std::strlen(vp.nickname);
    nickname = new char[nn_length + 1];
    std::strcpy(nickname, vp.nickname);
    year = vp.year;
}

VintagePort & VintagePort::operator=(const VintagePort & vp)
{
    delete nickname;
    Port::operator=(vp);

    int nn_length = std::strlen(vp.nickname);
    nickname = new char[nn_length + 1];
    std::strcpy(nickname, vp.nickname);
    year = vp.year;

    return *this;
}

void VintagePort::Show() const
{ 
    Port::Show();
    cout << "Nickname: " << nickname << endl;
    cout << "Year: " << year << endl;
}

ostream & operator<<(ostream & os, const VintagePort & vp)
{
    os << Port(vp);
    os << ", " << vp.nickname << ", " << vp.year;

    return os;
}

```



# 第十四章课后习题

```
// Chapter14.cpp
#include <iostream>
#include <array>
#include <string>
#include <ctime>
#include "Pairs.h"
#include "QueueTp.h"
#include "Worker.h"
#include "Person.h"
#include "emp.h"

// practice 1的时候include Wine.h
// 还要将Wine.cpp中的 #if 0 改成 #if 1
// 还要将Wine1.cpp中的 #if 1 改成 #if 0
//#include "Wine.h"

// practice 2的时候include Wine1.h
#include "Wine1.h"

using namespace std;

// practice 1 && practice 2
void p14_1(void)
{
    cout << "Enter name of wine: ";
    char lab[50];
    cin.getline(lab, 50);
    cout << "Enter number of years: ";
    int yrs;
    cin >> yrs;

    Wine holding(lab, yrs);
    holding.GetBottles();
    holding.Show();

    const int YRS = 3;
    int y[YRS] = { 1993, 1995, 1998 };
    int b[YRS] = { 48, 60, 72 };

    Wine more("Gushing Grape Red", YRS, y, b);
    more.Show();

    cout << "Total bottles for " << more.Label()
        << ": " << more.sum() << endl;

    cout << "Bye\n";
    return;
}


// practice 3
void p14_3(void)
{
    QueueTp<Worker> * Qworker = new QueueTp<Worker>(20);

    // 放在大括号里面是想看看Queue里面的item是否受原数据生命周期的影响的
    // 看来是不会受到影响
    {
        Worker w1("jimmy", 0);
        Worker w2("jimmy2", 1);
        if (Qworker->isempty())
        {
            cout << "Qworker is empty!\n";
        }
        Qworker->enqueue(w1);
        Qworker->enqueue(w2);
        cout << "Qworker count: " << Qworker->queuecount() << endl;

        Worker w3("hello", 2);
        Qworker->enqueue(w3);
        Qworker->enqueue(w2);
        cout << "Qworker count: " << Qworker->queuecount() << endl;
    }

    Worker temp;
    Qworker->dequeue(temp);
    cout << "Dequeue: " << endl;
    temp.Show();

    Qworker->dequeue(temp);
    cout << "Dequeue: " << endl;
    temp.Show();

    cout << "Qworker count: " << Qworker->queuecount() << endl;
}


// practice 4
// 勉勉强强用吧，感觉写得我自己都有点晕了
void p14_4(void)
{
    const int SIZE = 5;
    srand(time(0));

    Person * person[SIZE];
    int ct = 0;
    for ( ct = 0; ct < SIZE; ct++)
    {
        char choice;
        cout << "Enter the choice:\n"
            << "g: Gunslinger  t: BadDude  "
            << "a: PokerPlayer  q: quit\n";
        cin >> choice;

        while (strchr("pgatq", choice) == NULL)
        {
            cout << "Please enter a p, g, a, t, q: ";
            cin >> choice;
        }
        while (cin.get() != '\n')
        {
            continue;
        }

        if (choice == 'q')
        {
            break;
        }

        string fname;
        string lname;
        int nicks = 0;

        cout << "Enter first name: " << endl;
        getline(cin, fname);
        cout << "Enter last name: " << endl;
        getline(cin, lname);

        switch (choice)
        {
        case 'g':
            cout << "Enter nicks: " << endl;
            cin >> nicks;
            person[ct] = new Gunslinger(fname, lname, nicks);
            break;

        case 'a':
            person[ct] = new PokerPlayer(fname, lname);
            break;

        case 't':
            cout << "Enter nicks: " << endl;
            cin >> nicks;
            person[ct] = new BadDude(fname, lname, nicks);
            break;

        default:
            break;
        }
    }

    cout << "\nHere is your input:\n";
    int i = 0;
    for (i = 0; i < ct; i++)
    {
        cout << endl;
        person[i]->Show();
    }

    for (i = 0; i < ct; i++)
    {
        delete person[i];
    }

    cout << "Bye!\n";
    return;
}


// practice 5
void p14_5(void)
{
    employee em("Trip", "Harris", "Thumper");
    cout << em << endl;
    em.ShowAll();
    manager ma("Amorphias", "Spindragon", "Nuancer", 5);
    cout << ma << endl;
    ma.ShowAll();

    fink fi("Matt", "Oggs", "Oiler", "Juno Barr");
    cout << fi << endl;
    fi.ShowAll();
    highfink hf(ma, "Curly Kew");
    hf.ShowAll();
    cout << "Press a key for next phase:\n";
    cin.get();
    highfink hf2;
    hf2.SetAll();

    cout << "Using an abstr_emp * pointer:\n";
    abstr_emp * tri[4] = { &em, &fi, &hf, &hf2 };
    for (int i = 0; i < 4; i++)
    {
        tri[i]->ShowAll();
    }

    return;
}



int main(int argc, char **argv)
{
    p14_5();

    while (cin.get());

    return 0;
}

// emp.h
#pragma once
#include <iostream>
#include <string>

class abstr_emp
{
public:
    abstr_emp();
    abstr_emp(const std::string & fn, const std::string & ln,
        const std::string & j);
    virtual void ShowAll() const;
    virtual void SetAll();
    friend std::ostream &
        operator<<(std::ostream & os, const abstr_emp & e);
    virtual ~abstr_emp() = 0;

private:
    std::string fname;
    std::string lname;
    std::string job;
};

class employee : public abstr_emp
{
public:
    employee();
    employee(const std::string & fn, const std::string & ln,
        const std::string & j);
    virtual void ShowAll() const;
    virtual void SetAll();
};

class manager: virtual public abstr_emp
{
private:
    int inchargeof;

protected:
    int InChargeOf() const { return inchargeof; }
    int & InChargeOf() { return inchargeof; }

public:
    manager();
    manager(const std::string & fn, const std::string & ln,
        const std::string & j, int ico = 0);
    manager(const abstr_emp & e, int ico);
    manager(const manager & m);
    virtual void ShowAll() const;
    virtual void SetAll();
};

class fink : virtual public abstr_emp
{
private:
    std::string reportsto;
protected:
    const std::string ReportsTo() const { return reportsto; }
    std::string & ReportsTo() { return reportsto; }
public:
    fink();
    fink(const std::string & fn, const std::string & ln,
        const std::string & j, const std::string & rpo);
    fink(const abstr_emp & e, const std::string & rep);
    fink(const fink & e);
    virtual void ShowAll() const;
    virtual void SetAll();
};

class highfink : public manager, public fink
{
public:
    highfink();
    highfink(const std::string & fn, const std::string & ln,
        const std::string & j, const std::string & rpo,
        int ico);
    highfink(const abstr_emp & e, const std::string & rpo, int ico);
    highfink(const fink & f, int ico);
    highfink(const manager & m, const std::string & rpo);
    highfink(const highfink & h);
    virtual void ShowAll() const;
    virtual void SetAll();
};


// emp.cpp
#include "emp.h"

using namespace std;

abstr_emp::abstr_emp() : fname("None"), lname("None"), job("None")
{

}

abstr_emp::abstr_emp(const string & fn, const string & ln,
    const string & j)
{
    lname = ln;
    fname = fn;
    job = j;
}

void abstr_emp::ShowAll() const
{
    cout << "abstr_emp: " << endl;
    cout << "First Name: " << fname << endl;
    cout << "Last Name: " << lname << endl;
    cout << "job: " << job << endl;
}

void abstr_emp::SetAll()
{
    cout << "Enter First name: ";
    getline(cin, fname);
    cout << "Enter Last name: ";
    getline(cin, lname);
    cout << "Enter job: ";
    getline(cin, job);
}

std::ostream & operator<<(std::ostream & os, const abstr_emp & e)
{
    cout << e.fname << ", " << e.lname ;
    return os;
}

abstr_emp::~abstr_emp()
{

}

employee::employee() : abstr_emp()
{

}

employee::employee(const string & fn, const string & ln,
    const string & j) : abstr_emp(fn, ln, j)
{

}

void employee::ShowAll() const
{
    cout << "employee: " << endl;
    abstr_emp::ShowAll();
}

void employee::SetAll()
{
    abstr_emp::SetAll();
}

manager::manager() : abstr_emp(), inchargeof(-1)
{

}

manager::manager(const string & fn, const string & ln,
    const string & j, int ico) : abstr_emp(fn, ln, j), inchargeof(ico)
{

}

manager::manager(const abstr_emp & e, int ico) :
    abstr_emp(e), inchargeof(ico)
{

}

manager::manager(const manager & m) : abstr_emp(m)
{
    inchargeof = m.inchargeof;
}

void manager::ShowAll() const
{
    cout << "manager: " << endl;
    abstr_emp::ShowAll();
    cout << "inchargeof: " << inchargeof << endl;
}

void manager::SetAll()
{
    abstr_emp::SetAll();
    cout << "Enter inchargeof: ";
    cin >> inchargeof;
    while (cin.get() != '\n')
    {
        continue;
    }
}

fink::fink() : abstr_emp(), reportsto("None")
{

}

fink::fink(const string & fn, const string & ln,
    const string & j, const string & rpo) : abstr_emp(fn, ln, j), reportsto(rpo)
{

}

fink::fink(const abstr_emp & e, const string & rpo) : abstr_emp(e), reportsto(rpo)
{

}

fink::fink(const fink & e) : abstr_emp(e)
{
    reportsto = e.reportsto;
}

void fink::ShowAll() const
{
    cout << "fink: " << endl;
    abstr_emp::ShowAll();
    cout << "reportsTo: " << reportsto << endl;
}

void fink::SetAll()
{
    abstr_emp::SetAll();
    cout << "Enter reportsTo: ";
    getline(cin, reportsto);
}

highfink::highfink() : abstr_emp(), manager(), fink()
{

}

highfink::highfink(const string & fn, const string & ln,
    const string & j, const string & rpo, int ico) :
    abstr_emp(fn, ln, j), manager(fn, ln, j, ico),
    fink(fn, ln, j, rpo)
{

}

highfink::highfink(const abstr_emp & e, const string & rpo, int ico) :
    abstr_emp(e), manager(e, ico), fink(e, rpo)
{

}

highfink::highfink(const fink & f, int ico) :
    abstr_emp(f), manager(f, ico), fink(f)
{

}

highfink::highfink(const manager & m, const string & rpo) :
    abstr_emp(m), manager(m), fink(m, rpo)
{

}

highfink::highfink(const highfink & h) :
    abstr_emp(h), manager(h), fink(h)
{

}

void highfink::ShowAll() const
{
    cout << "highfink: " << endl;
    abstr_emp::ShowAll();
    cout << "inchargeof: " << manager::InChargeOf() << endl;
    cout << "reportsTo: " << fink::ReportsTo() << endl;
}

void highfink::SetAll()
{
    manager::SetAll();
    cout << "Enter reportsTo: ";
    getline(cin, fink::ReportsTo());
}


// Pairs.h
#pragma once
#include <iostream>
#include <string>

template<class T1, class T2>
class Pair {
private:
    T1 a;
    T2 b;

public:
    T1 & first();
    T2 & second();
    T1 first() const { return a; }
    T2 second() const { return b; }
    Pair(const T1 & aval, const T2 & bval) : a(aval), b(bval) {}
    Pair() {}
};

template<class T1, class T2>
T1 & Pair<T1, T2>::first()
{
    return a;
}

template<class T1, class T2>
T2 & Pair<T1, T2>::second()
{
    return b;
}


// Person.h
#pragma once
#include <string>

class Person
{
private:
    std::string firstName;
    std::string lastName;

public:
    Person() : firstName("None"), lastName("None") {}
    Person(const std::string fname, const std::string lname) :
        firstName(fname), lastName(lname) {}
    Person(const Person & p);
    virtual ~Person() = 0;
    virtual void Show() const = 0;
};

class Gunslinger : public virtual Person
{
private:
    int nicks;

public:
    Gunslinger(int d = 0) : Person(), nicks(d) {}
    Gunslinger(const std::string fname, const std::string lname, int d) :
        Person(fname, lname), nicks(d) {}
    Gunslinger(const Person & p, int d) : Person(p), nicks(d) {}
    Gunslinger(const Gunslinger & g);
    ~Gunslinger();
    void Show() const;
    double Draw() const;
};

class PokerPlayer : public virtual Person
{
public:
    int Draw() const;
    void Show() const;
    PokerPlayer() : Person() {}
    PokerPlayer(const Person & p) : Person(p) {}
    PokerPlayer(const std::string fname, const std::string lname) :
        Person(fname, lname) {}
    PokerPlayer(const PokerPlayer & p) : Person(p) {}
    ~PokerPlayer();
};

class BadDude : public Gunslinger, public PokerPlayer
{
public:
    BadDude() {}
    BadDude(const std::string fname, const std::string lname, int d) :
        Person(fname, lname), Gunslinger(fname, lname, d), PokerPlayer(fname, lname) {}
    BadDude(const Person & p, int d) :
        Person(p), Gunslinger(p, d), PokerPlayer(p) {}
    BadDude(const Gunslinger & g) :
        Person(g), Gunslinger(g), PokerPlayer(g) {}
    BadDude(const PokerPlayer & p, int d) :
        Person(p), Gunslinger(p, d), PokerPlayer(p) {}
    ~BadDude();

    void Show() const;
    double Gdraw();
    int Cdraw();
};


// Person.cpp
#include "Person.h"
#include <iostream>
#include <ctime>
#include <cmath>

using std::cout;
using std::endl;

Person::Person(const Person & p)
{
    firstName = p.firstName;
    lastName = p.lastName;
}

Person::~Person()
{
}

void Person::Show() const
{
    cout << "First Name: " << firstName << endl;
    cout << "Last Name: " << lastName << endl;
}

Gunslinger::Gunslinger(const Gunslinger & g) : Person(g)
{
    nicks = g.nicks;
}

double Gunslinger::Draw() const
{
    return (double)(rand() % 10) / 10;
}

void Gunslinger::Show() const
{
    Person::Show();
    cout << "Nicks: " << nicks << endl;
    cout << "Pull the gun time: " << Draw() << endl;
}

Gunslinger::~Gunslinger()
{

}

int PokerPlayer::Draw() const
{
    return (rand() % 52);
}

void PokerPlayer::Show() const
{
    Person::Show();
}

PokerPlayer::~PokerPlayer()
{

}


void BadDude::Show() const
{
    Gunslinger::Show();
    cout << "Next Card: " << PokerPlayer::Draw() << endl;
}

double BadDude::Gdraw()
{
    return Gunslinger::Draw();
}

int BadDude::Cdraw()
{
    return PokerPlayer::Draw();
}

BadDude::~BadDude()
{

}


// QueueTp.h
#pragma once

// 这个模板类的内容是按照程序清单12.10来写的
// 还要注意模板类定义和实现都要放在头文件中，不然编译会报错
template <class T>
class QueueTp
{
private:
    struct Node { T item; struct Node * next; };
    enum { Q_SIZE = 10 };
    Node * front;
    Node * rear;
    int items;
    const int qsize;
    QueueTp(const QueueTp & q) : qsize(0) {}
    QueueTp & operator=(const QueueTp & q) { return *this; }

public:
    QueueTp(int qs = Q_SIZE);
    ~QueueTp();
    bool isempty() const;
    bool isfull() const;
    int queuecount() const;
    bool enqueue(const T & item);
    bool dequeue(T & item);
};

template <class T>
QueueTp<T>::QueueTp(int qs) : qsize(qs)
{
    front = rear = NULL;
    items = 0;
}

template <class T>
QueueTp<T>::~QueueTp()
{
    Node * temp;
    while (front != NULL)
    {
        temp = front;
        front = front->next;
        delete temp;
    }
}

template <class T>
bool QueueTp<T>::isempty() const
{
    return items == 0;
}

template <class T>
bool QueueTp<T>::isfull() const
{
    return items == qsize;
}

template <class T>
int QueueTp<T>::queuecount() const
{
    return items;
}

template <class T>
bool QueueTp<T>::enqueue(const T & item)
{
    if (isfull())
    {
        return false;
    }
    Node * add = new Node;
    add->item = item;
    add->next = NULL;
    items++;
    if (front == NULL)
    {
        front = add;
    }
    else
        rear->next = add;
    rear = add;
    return true;
}

template <class T>
bool QueueTp<T>::dequeue(T & item)
{
    if (front == NULL)
    {
        return false;
    }
    item = front->item;
    items--;
    Node * temp = front;
    front = front->next;
    delete temp;
    if (items == 0)
    {
        rear = NULL;
    }

    return true;
}


// Wine.h
#pragma once
#include <iostream>
#include <array>
#include <string>
#include <valarray>
#include "Pairs.h"

typedef std::valarray<int> ArrayInt;
typedef Pair<ArrayInt, ArrayInt> PairArray;

class Wine
{
private:
    int year;
    std::string label;
    PairArray pArray;

public:
    Wine(const char * l, int y, const int yr[], const int bot[]);
    Wine(const char * l, int y);
    ~Wine();
    void GetBottles();
    void Show();
    std::string & Label();
    int sum();
};

// Wine.cpp
#include "Wine.h"

#if 0

Wine::Wine(const char * l, int y, const int yr[], const int bot[])
{
    pArray = PairArray(ArrayInt(y), ArrayInt(y));
    label = l;
    year = y;
    for (int  i = 0; i < y; i++)
    {
        pArray.first()[i] = yr[i];
        pArray.second()[i] = bot[i];
    }
}

Wine::Wine(const char * l, int y)
{
    pArray = PairArray(ArrayInt(y), ArrayInt(y));
    label = l;
    year = y;
    for (int i = 0; i < y; i++)
    {
        pArray.first()[i] = 0;
        pArray.second()[i] = 0;
    }
}


Wine::~Wine()
{
}

void Wine::GetBottles()
{
    std::cout << "Enter " << label << " data for " << year << " year(s):" << std::endl;
    for (int i = 0; i < year; i++)
    {
        std::cout << "Enter year: ";
        std::cin >> pArray.first()[i];
        std::cout << "Enter bootles for that year: ";
        std::cin >> pArray.second()[i];
    }
}

void Wine::Show()
{
    std::cout << "Wine: " << label << std::endl;
    std::cout << "  Year    Bottles" << std::endl;
    for (int i = 0; i < year; i++)
    {
        std::cout << "  " << pArray.first()[i] << " " << pArray.second()[i] << std::endl;
    }
}

std::string & Wine::Label()
{
    return label;
}

int Wine::sum()
{
    int sum = 0;
    for (int i = 0; i < year; i++)
    {
        sum += pArray.second()[i];
    }

    return sum;
}

#endif


// Wine1.h
#pragma once
#include <iostream>
#include <string>
#include <valarray>
#include "Pairs.h"

typedef std::valarray<int> ArrayInt;
typedef Pair<ArrayInt, ArrayInt> PairArray;

class Wine : public PairArray, public std::string
{
private:
    int year;

public:
    Wine(const char * l, int y, const int yr[], const int bot[]);
    Wine(const char * l, int y);
    ~Wine();
    void GetBottles();
    void Show();
    std::string & Label();
    int sum();
};


// Wine1.cpp
#include "Wine1.h"

#if 1

Wine::Wine(const char * l, int y, const int yr[], const int bot[]) :
    std::string(l), PairArray(ArrayInt(y), ArrayInt(y))
{
    year = y;
    for (int i = 0; i < year; i++)
    {
        PairArray::first()[i] = yr[i];
        PairArray::second()[i] = bot[i];
    }
}

Wine::Wine(const char * l, int y) :
    std::string(l), PairArray(ArrayInt(y), ArrayInt(y))
{
    year = y;
    for (int i = 0; i < year; i++)
    {
        PairArray::first()[i] = 0;
        PairArray::second()[i] = 0;
    }
}

Wine::~Wine()
{

}

void Wine::GetBottles()
{
    std::cout << "Enter " << (const std::string)*this << " data for " << year
        << " year(s): " << std::endl;
    for (int i = 0; i < year; i++)
    {
        std::cout << "Enter year: ";
        std::cin >> PairArray::first()[i];
        std::cout << "Enter bootles for that year: ";
        std::cin >> PairArray::second()[i];
    }
}

void Wine::Show()
{
    std::cout << "Wine: " << (const std::string)*this << std::endl;
    std::cout << "  Year    Bootles" << std::endl;
    for (int i = 0; i < year; i++)
    {
        std::cout << "  " << PairArray::first()[i] << " "
            << PairArray::second()[i] << std::endl;
    }
}

std::string & Wine::Label()
{
    return (std::string &)*this;
}

int Wine::sum()
{
    int sum = 0;
    for (int i = 0; i < year; i++)
    {
        sum += PairArray::second()[i];
    }

    return sum;
}

#endif


// Worker.h
#pragma once
#include <string>


class Worker
{
private:
    std::string fullname;
    long id;

public:
    Worker() : fullname("no one"), id(0L) {}
    Worker(const std::string & s, long n)
        : fullname(s), id(n) {}
    virtual ~Worker();
    virtual void Set();
    virtual void Show() const;
};

// Worker.cpp
#include "Worker.h"
#include <iostream>

using std::cout;
using std::endl;
using std::cin;


void Worker::Set()
{
    cout << "Enter worker's name: ";
    std::getline(cin, fullname);
    cout << "Enter worker's ID: ";
    cin >> id;
    while (cin.get() != '\n')
    {
        continue;
    }
}

void Worker::Show() const
{
    cout << "Name: " << fullname << "\n";
    cout << "Employee ID: " << id << "\n";
}


Worker::~Worker()
{
}

```



# 第十五章课后习题

```
#include <iostream>
#include <string>
#include <cstring>
#include <set>
#include <map>
#include <iterator>
#include <vector>
#include <fstream>
#include <ctime>
#include <list>
#include <queue>
#include <memory>

using namespace std;

// practice 1
bool Palindrome(string input)
{
    string::iterator forware = input.begin();
    string::reverse_iterator backware = input.rbegin();
    int input_length = input.length();

    for (int i = 0; i < input_length / 2; i++)
    {
        if (*forware == *backware)
        {
            forware++;
            backware++;
        }
        else
            return false;
    }

    return true;
}
void p16_1(void)
{
    string input;
    cout << "Enter the string: ";
    getline(cin, input);

    if (Palindrome(input))
    {
        cout << input << " is a palindrome." << endl;
    }
    else
    {
        cout << input << " is not a palindrome." << endl;
    }
}


// practice 2
void p16_2(void)
{
    string input;
    string temp;

    cout << "Enter string: ";
    getline(cin, input);
    temp = input;

    for (string::iterator it = input.begin(); it != input.end();)
    {
        if (!isalpha(*it))
        {
            it = input.erase(it);
            continue;
        }
        else
        {
            *it = tolower(*it);
            it++;
        }
    }

    if (Palindrome(input))
    {
        cout << input << " is a palindrome." << endl;
    }
    else
    {
        cout << input << " is not a palindrome." << endl;
    }
}


// practice 3
const string FileName = "word.txt";
void p16_3(void)
{
    vector<string> wordlist;
    ifstream in;
    in.open(FileName.c_str());
    string inword;


    while (in >> inword)
    {
        wordlist.push_back(inword);
    }

    srand(time(0));
    char play;
    cout << "Will you play a word game? <y/n> ";
    cin >> play;
    play = tolower(play);
    while (play == 'y')
    {
        string target = wordlist[rand() % wordlist.size()];
        int length = target.length();
        string attemp(length, '-');
        string badchars;
        int guesses = 6;
        cout << "Guess my scret word. It has " << length
            << " letters, and you guess\n"
            << "one letter at a time. You get " << guesses
            << " wrong guesses.\n";
        cout << "Your word: " << attemp << endl;
        while (guesses > 0 && attemp != target)
        {
            char letter;
            cout << "Guess a letter: ";
            cin >> letter;
            if (badchars.find(letter) != string::npos
                || attemp.find(letter) != string::npos)
            {
                cout << "You already gueesed that. Try again.\n";
                continue;
            }

            int loc = target.find(letter);
            if (loc == string::npos)
            {
                cout << "Oh, bad guess!\n";
                --guesses;
                badchars += letter;
            }
            else
            {
                cout << "Good guess!\n";
                attemp[loc] = letter;
                loc = target.find(letter, loc + 1);
                while (loc != string::npos)
                {
                    attemp[loc] = letter;
                    loc = target.find(letter, loc + 1);
                }
            }

            cout << "You word: " << attemp << endl;

            if (attemp != target)
            {
                if (badchars.length() > 0)
                    cout << "Bad choices: " << badchars << endl;
                cout << guesses << " bad guesses left\n";
            }
        }
        if (guesses > 0)
        {
            cout << "That's right!\n";
        }
        else
        {
            cout << "Sorry, the word is " << target << ".\n";
        }

        cout << "Will you play anther? <y/n> ";
        cin >> play;
        play = tolower(play);
    }

    cout << "Bye\n";

    return;
}


// practice 4
int reduce(long ar[], int n)
{
    list<long> lar(ar, ar + n);
    lar.sort();
    lar.unique();
    int lar_length = 0;
    for (list<long>::iterator it = lar.begin(); it != lar.end(); it++)
    {
        *(ar + lar_length) = *it;
        lar_length++;
    }

    for (int i = lar_length; i < n; i++)
    {
        *(ar + lar_length) = 0;
    }

    return lar_length;

}
void p16_4(void)
{
    long ar[] = { 12, 2, 13, 12, 2, 55, 32, 44, 32, 100, 32, 12 };
    int ar_length = 12;
    int reduce_length = 0;

    cout << "original array length: " << ar_length << endl;
    cout << "original array: ";
    for (int i = 0; i < ar_length; i++)
    {
        cout << ar[i] << " " ;
    }
    cout << endl;

    reduce_length = reduce(ar, ar_length);
    cout << "After reduce, array length: " << reduce_length << endl;
    cout << "After reduce, array: ";
    for (int i = 0; i < reduce_length; i++)
    {
        cout << ar[i] << " " ;
    }
    cout << endl;
}


// practice 5
template <class T>
int Treduce(T ar[], int n)
{
    list<T> lar(ar, ar + n);
    lar.sort();
    lar.unique();
    int lar_length = 0;
    for (list<T>::iterator it = lar.begin(); it != lar.end(); it++)
    {
        *(ar + lar_length) = *it;
        lar_length++;
    }

    return lar_length;
}
void p16_5(void)
{
    // long
    long ar[] = { 12, 2, 13, 12, 2, 55, 32, 44, 32, 100, 32, 12 };
    int ar_length = 12;
    int reduce_length = 0;

    cout << "original array length: " << ar_length << endl;
    cout << "original array: ";
    for (int i = 0; i < ar_length; i++)
    {
        cout << ar[i] << " ";
    }
    cout << endl;

    reduce_length = Treduce(ar, ar_length);
    cout << "After reduce, array length: " << reduce_length << endl;
    cout << "After reduce, array: ";
    for (int i = 0; i < reduce_length; i++)
    {
        cout << ar[i] << " ";
    }
    cout << endl;

    cout << endl;
    // string
    string sar[] = { "hello", "jimmy", "array", "hehe", "jimmy", "hehe" };
    int sar_length = 6;
    cout << "original string array length: " << sar_length << endl;
    cout << "original string array: ";
    for (int i = 0; i < sar_length; i++)
    {
        cout << sar[i] << " ";
    }
    cout << endl;

    reduce_length = Treduce(sar, sar_length);
    cout << "After reduce, array length: " << reduce_length << endl;
    cout << "After reduce, array: ";
    for (int i = 0; i < reduce_length; i++)
    {
        cout << sar[i] << " ";
    }
    cout << endl;
}


// practice 6
class Customer {
private:
    long arrive;
    int processtime;
public:
    Customer() { arrive = processtime = 0; }
    void set(long when) {
        processtime = rand() % 3 + 1;
        arrive = when;
    }
    long when() const { return arrive; }
    int ptime() const { return processtime; }
};
const int MIN_PER_HR = 60;
bool newcustomer(double x)
{
    return (rand() * x / RAND_MAX < 1);
}
void p16_6(void)
{
    srand(time(0));
    cout << "Case study: Band of Heather Automatic Teller\n";
    cout << "Enter maximum size of queue: ";
    int qs;
    cin >> qs;
    queue<Customer> line;

    cout << "Enter the number of simulation hours: ";
    int hours;
    cin >> hours;
    long cyclelimit = MIN_PER_HR * hours;

    cout << "Enter the average number of customer per hours: ";
    double perhour;
    cin >> perhour;
    double min_per_cust;
    min_per_cust = MIN_PER_HR / perhour;

    Customer temp;
    long turnaways = 0;
    long customers = 0;
    long served = 0;
    long sum_line = 0;
    int wait_time = 0;
    long line_wait = 0;

    for (int cycle = 0; cycle < cyclelimit; cycle++)
    {
        if (newcustomer(min_per_cust))
        {
            if (qs == line.size())
                turnaways++;
            else
            {
                customers++;
                temp.set(cycle);
                line.push(temp);
            }
        }
        if (wait_time <= 0 && !line.empty())
        {
            temp = line.front();
            wait_time = temp.ptime();
            line_wait += cycle - temp.when();
            served++;
            line.pop();
        }
        if (wait_time > 0)
        {
            wait_time--;
        }
        sum_line += line.size();
    }

    if (customers > 0)
    {
        cout << "customers accepted: " << customers << endl;
        cout << "  customers served: " << served << endl;
        cout << "         turnaways: " << turnaways << endl;
        cout << "average queue size: ";
        cout.precision(2);
        cout.setf(ios_base::fixed, ios_base::floatfield);
        cout << (double)sum_line / cyclelimit << endl;
        cout << " average wait time: "
            << (double)line_wait / served << " minutes\n";
    }
    else
        cout << "No customers!\n";

    cout << "Done!\n";

    return;

}


// practice 7
vector<int> Lotto(int total, int choice)
{
    vector<int> ar(total);
    for (int i = 0; i < total; i++)
    {
        ar[i] = i + 1;
    }

    random_shuffle(ar.begin(), ar.end());

    vector<int> car(choice);
    for (int i = 0; i < choice; i++)
    {
        car[i] = ar[i];
    }
    sort(car.begin(), car.end());
    return car;
}
void p16_7(void)
{
    srand(time(0));
    vector<int> winners;
    winners = Lotto(51, 6);
    cout << "Winners' number: ";
    for (size_t i = 0; i < 6; i++)
    {
        cout << winners[i] << " ";
    }
    cout << endl;

    return;
}


// practice 8
void p16_8(void)
{
    set<string> Mat_friend;
    set<string> Pat_friend;
    set<string> union_friend;
    string input;

    cout << "Mat friend: ";
    while (cin >> input)
    {
        Mat_friend.insert(input);
        if (cin.get() == '\n')
        {
            break;
        }
    }
    copy(Mat_friend.begin(), Mat_friend.end(), ostream_iterator<string, char>(cout, " "));
    cout << endl;

    cout << "Pat friend: ";
    while (cin >> input)
    {
        Pat_friend.insert(input);
        Mat_friend.insert(input);
        if (cin.get() == '\n')
        {
            break;
        }
    }
    copy(Pat_friend.begin(), Pat_friend.end(), ostream_iterator<string, char>(cout, " "));
    cout << endl;

    set_union(Mat_friend.begin(), Mat_friend.end(), Pat_friend.begin(), Pat_friend.end(), insert_iterator<set<string> >(union_friend, union_friend.begin()));

    cout << "All friends: ";
    copy(union_friend.begin(), union_friend.end(), ostream_iterator<string, char>(cout, " "));
    cout << endl;
}


// practice 9
const long VEC_SIZE = 1000000;  // 这个量级i5-3320M大概用了一分多钟
void p16_9(void)
{
    cout << "-----start-----" << endl;
    srand(time(0));
    vector<int> vi0(VEC_SIZE);
    for (long i = 0; i < VEC_SIZE; i++)
    {
        vi0[i] = rand() % 10000 + 1;
    }

    vector<int> vi(vi0);
    list<int> li(vi0.begin(), vi0.end());

    clock_t vector_start = clock();
    sort(vi.begin(), vi.end());
    clock_t vector_end = clock();

    clock_t list_start = clock();
    li.sort();
    clock_t list_end = clock();

    copy(vi0.begin(), vi0.end(), li.begin());
    clock_t copy_sort_time_start = clock();
    copy(li.begin(), li.end(), vi.begin());
    sort(vi.begin(), vi.end());
    copy(vi.begin(), vi.end(), li.begin());
    clock_t copy_sort_time_end = clock();

    cout << "test vector size: " << VEC_SIZE << endl;
    cout << "vector sort time: " << (double)(vector_end - vector_start) / CLOCKS_PER_SEC << endl;
    cout << "list sort time: " << (double)(list_end - list_start) / CLOCKS_PER_SEC << endl;
    cout << "list copy sort time: " << (double)(copy_sort_time_end - copy_sort_time_start) / CLOCKS_PER_SEC << endl;
    cout << "-----end-----" << endl;
}


// practice 10
struct Review
{
    string title;
    int rating;
    double price;
};
bool FillRevice(Review & rr)
{
    cout << "Enter book title (quit to quit): ";
    getline(cin, rr.title);
    if (rr.title == "quit")
    {
        return false;
    }
    cout << "Enter book rating: ";
    cin >> rr.rating;
    if (!cin)
    {
        return false;
    }

    cout << "Enter book price: ";
    cin >> rr.price;
    if (!cin)
    {
        return false;
    }
    while (cin.get() != '\n')
        continue;

    return true;
}
void ShowReview(const shared_ptr<Review> & rr)
{
    cout << rr->rating << "\t" << rr->title << "\t" << rr->price << endl;
}
bool operator<(const shared_ptr<Review> & r1, const shared_ptr<Review> & r2)
{
    if (r1->title < r2->title)
    {
        return true;
    }
    else if (r1->title == r2->title && r1->rating < r2->rating)
        return true;
    else
        return false;
}
bool worseThan(const shared_ptr<Review> & r1, const shared_ptr<Review> & r2)
{
    if (r1->rating < r2->rating)
        return true;
    else
        return false;
}
bool betterThan(const shared_ptr<Review> & r1, const shared_ptr<Review> & r2)
{
    if (r1->rating > r2->rating)
        return true;
    else
        return false;
}
bool expensiveThan(const shared_ptr<Review> & r1, const shared_ptr<Review> & r2)
{
    if (r1->price > r2->price)
        return true;
    else
        return false;
}
bool cheaperThan(const shared_ptr<Review> & r1, const shared_ptr<Review> & r2)
{
    if (r1->price < r2->price)
        return true;
    else
        return false;
}
void p16_10(void)
{
    vector<shared_ptr<Review> > books;
    Review temp;
    while (FillRevice(temp))
    {
        shared_ptr<Review> in(new Review(temp));
        books.push_back(in);
    }
    cout << "Enter your choice: " << endl;
    cout << "0：按原始顺序显示， 1：按字母表顺序显示， 2：按评级升序显示" << endl;
    cout << "3：按评级降序显示， 4：按价格升序显示  ， 5：按价格降序显示" << endl;
    cout << "6：退出" << endl;
    int choice = 0;
    vector<shared_ptr<Review> > original_books(books.begin(), books.end());
    while (cin >> choice && choice != 6)
    {
        switch (choice)
        {
        case 0:
            copy(original_books.begin(), original_books.end(), books.begin());
            break;

        case 1:
            sort(books.begin(), books.end());
            break;

        case 2:
            sort(books.begin(), books.end(), worseThan);
            break;

        case 3:
            sort(books.begin(), books.end(), betterThan);
            break;

        case 4:
            sort(books.begin(), books.end(), cheaperThan);
            break;

        case 5:
            sort(books.begin(), books.end(), expensiveThan);
            break;

        default:
            break;
        }

        for_each(books.begin(), books.end(), ShowReview);
    }

    return;
}




int main(int argc, char ** argv)
{
    p16_10();

    while (cin.get());

    return 0;
}

```



