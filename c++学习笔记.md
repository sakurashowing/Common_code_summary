# 1、sizeof 和 strlen区别

1、sizeof 操作符的结果类型是 size_t，它在头文件中 typedef 为 unsigned int 类型。该类型保证能容纳实现所建立的最大对象的字节大小。

2、sizeof 是运算符，strlen 是函数。

3、sizeof 可以用类型做参数，**strlen** 只能用 **char\*** 做参数，且必须是以 **\0** 结尾的。

sizeof 还可以用函数做参数，比如：

```C++
short f();
printf("%d\n", sizeof(f()));
```

输出的结果是 sizeof(short)，即 2。

4、数组做 **sizeof** 的参数不退化，传递给 **strlen** 就退化为指针了。

5、大部分编译程序在编译的时候就把 **sizeof** 计算过了，是类型或是变量的长度，这就是 **sizeof(x)** 可以用来定义数组维数的原因。

```C++
char str[20]="0123456789";
int a=strlen(str); // a=10;
int b=sizeof(str); // 而 b=20;
```

6、strlen 的结果要在运行的时候才能计算出来，是用来计算字符串的长度，不是类型占内存的大小。

7、sizeof 后如果是类型必须加括弧，如果是变量名可以不加括弧。这是因为 **sizeof** 是个操作符不是个函数。

8、当适用一个结构类型或变量时， sizeof 返回实际的大小；当适用一静态地空间数组， sizeof 归还全部数组的尺寸；sizeof 操作符不能返回动态地被分派了的数组或外部的数组的尺寸。

数组作为参数传给函数时传的是指针而不是数组，传递的是数组的首地址， 如：

```
fun(char [8])
fun(char [])
```

都等价于

```
fun(char *) 
```

在 C++ 里参数传递数组永远都是传递指向数组首元素的指针，编译器不知道数组的大小。

如果想在函数内知道数组的大小， 需要这样做：

进入函数后用memcpy拷贝出来，长度由另一个形参传进去

```
fun(unsiged char *p1, int len)
{
    unsigned char* buf = new unsigned char[len+1]
    memcpy(buf, p1, len);
}
```

看了上面的详细解释，发现两者的使用还是有区别的，从这个例子可以看得很清楚：

```
char str[20]="0123456789";
int a=strlen(str);         // a=10; >>>> strlen 计算字符串的长度，以结束符 0x00 为字符串结束。
int b=sizeof(str);         // 而 b=20; >>>> sizeof 计算的则是分配的数组 str[20] 所占的内存空间的大小，不受里面存储的内容改变。  
```

上面是对静态数组处理的结果，如果是对指针，结果就不一样了。

```
char* ss = "0123456789";
sizeof(ss) 结果 4 ＝＝＝》ss 是指向字符串常量的字符指针，sizeof 获得的是一个指针的之所占的空间,应该是长整型的，所以是 4。
sizeof(*ss) 结果 1 ＝＝＝》*ss 是第一个字符 其实就是获得了字符串的第一位 '0' 所占的内存空间，是 char 类型的，占了 1 位
strlen(ss)= 10      ＝＝＝》 如果要获得这个字符串的长度，则一定要使用 strlen。strlen 用来求字符串的长度；而 sizeof 是用来求指定变量或者变量类型等所占内存大小。
```

# 2、指针

```c++
#include <iostream>
using namespace std;
int main ()
{
    int  var = 20; // 实际变量的声明
    int  *ip;        // 指针变量的声明 
    //ip = &var;       // 在指针变量中存储 var 的地址
    *ip=var;
    cout << "Value of var variable: ";
    cout << var << endl;
    // 输出在指针变量中存储的地址
    cout << "Address stored in ip variable: ";
    cout << ip << endl;
    // 访问指针中地址的值
    cout << "Value of *ip variable: ";
    cout << *ip << endl;
    
    return 0;
}
```

```
Value of var variable: 20
Address stored in ip variable: 0x7ffeefbff5a8
Value of *ip variable: 20
```

&：获取实际变量的**地址**

*：声明指针变量、返回位于操作数所指定地址的**变量的值**。

1、内存四区

代码区——代码

全局区——全局的常量例如字符串常量

栈区——系统自动开辟，自动释放不大

堆区——动态开辟的内存，手动开辟释放，很大

2、地址

把内存以单个字节为单位分开，对每个字节编号，这个编号就是地址

编号是连续的、唯一的、&取地址运算符为单目运算符，优先级仅次于（）[]、结合性从右往左

3、首地址

一段空间中第一个存储单元的地址（存储单元）

4、指针变量

专门存放地址的变量，地址是编号，它是一种数据

```
ing a;//整数
char c;//字符
地址。指针变量
```

指针变量的定义：数据类型 *变量名

```
int *p; //定义一个指针变量p，存的是地址！int 表明指针指向的数据类型
```

指针变量的赋值：

```
p=&a; //p保存a的地址 p指向a
```

指针变量的引用：

访问a这个int变量

1、使用变量名

```
printf（“a=%d\n”,a）;
```

2、指针访问:*指针变量 *是取值运算符。返回某个地址的值，单目运算符，从右到左

```
printf("p指向的内存的值：%d\n",*p);
```

注意：在定义指针变量时 int *p; *只是表明p是一个指针变量

​			非定义时， *p；取值p指向的内存值

野指针：不能明确指向的指针变量，很危险

```
int *p;  //p里面保存的地址不确定的，p指向不明确的
int *p=NULL; //可以避免野指针的危险
```

空指针：void* 转换成其他的数据类型

指针变量的运算： + -  ++ --  指针偏移，去访问地址旁边的一些内存

指针变量的加减，以指针所指向的类型空间单位进行偏移。

```
char *p; //p char 1 p+1 1B
int *p1; //  int  4 p1+1 4B
double *p2; // double 8  p2+1 8B
```

**首地址：一段空间中第一个存储单元的地址（存储单元）**

**指针变量的加减，以指针所指向的类型空间单位进行偏移。**

一位数组与指针：

1、定义一个一位数组，**数组名是这个数组的首地址**

```
int a[5];
a指向a[0],a[0] int元素， a的类型就是int*
a 这个地址指向a[0] int元素  int* 4     a+1 增加4个字节
&a 这个地址指向整个数组  int(*)[5]      &a+1 增加20个字节
```

2、访问数组元素：

```
int a[5]={1,2,3,4,5};
//下标法
for(int i=0;i<5;i++)
{
	printf("%d",a[i]);
}
//指针法
int *p=a;  //p指向a[0] p+1指向a[1]
for(int i=0;i<5,i++)
{
	print ("%d",*(p+i)); //或*p++  或*(a+i)  a是数组名 不可以改变的 不能用*a++
}
//*(p+i)  *p++ *和++优先级相同  从右往左  所以不用*(p++)
//*(p+i) p值不变  *p++ p值会变
```

二维数组与指针：数组名是这个数组的首地址,首地址：一段空间中第一个存储单元的地址（存储单元）

```
int a[3][4]={1,2,3,4,5,6,7,8,9,10.11.12};
```

数组名:a

```
a是指向a[0]这个一维数组，a的类型是 int*[4];   a+1  增加16个字节
a[0]指向a[0][0]这个元素，a[0]的类型是  int*   a[0]+1  增加4个字节
&a; int(*)[3][4]   &a+1  增加3*4*4个字节
```

```
int a[2][3][4];
a指向二维数组
a[0]指向一位数组
a[0][0]指向一个元素
```

```
m行n列的元素
	a[m][n]
	*(a[m]+n)
	*(*(a+m)+n)
```

```
int a[3][4]
a 	 int(*)[4]
&a 	 int(*)[3][4]
a[0] int*
a[0][0]  int
```

# 3、红包分配

```
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <math.h>

using namespace std;

int main()
{
    int i, number;
    int best;//手气最佳
    float total;
    
    cout << "请输入红包金额：";
    cin >> total;
    cout << "请输入抢红包人数：";
    cin >> number;
    /* 生成随机数 */
    // 设置种子
    srand((unsigned)time(NULL));
    float a[1024];//保存每个人的随机数。最多支持1024个人抢红包。
    float b[1024];//保存每个人获得的红包金额。
    float suma = 0;//随机数总和。
    float sumb = 0;//红包总和。
    int max = 0;
    for (i = 0; i < number; i++)
    {
        // 生成实际的随机数
        a[i] = rand() % 100;
        
        if (a[i] > max){
            max = a[i];
            best = i;//获取手气最佳
        }
        suma += a[i];
    }
    
    for (i = 0; i < number - 1; i++)
    {
        b[i] = a[i] / suma * total;//按照随机数计算每个人实际获得的金额
        sumb += round(b[i] * 100) / 100.0;//将红包金额保留两位小数
        //输出信息
        cout << "第" << setiosflags(ios::right)<< setw(3) << i + 1 <<
        "个人的红包是:" << setiosflags(ios::right) << setw(6) <<
        setiosflags(ios::fixed) << setprecision(2) <<
        round(b[i] * 100) / 100.0 ;
        if (best == i){
            cout << "(手气最佳)" << endl;
        }
        else {
            
            cout << endl;
        }
    }
    //最后一人的红包金额等于总金额减去前面的金额。
    cout << "第" << setiosflags(ios::right)<<
    setw(3) << number << "个人的红包是:" <<
    setiosflags(ios::right) << setw(6) << setiosflags(ios::fixed) <<
    setprecision(2) << round((total - sumb) * 100) / 100.0;
    if (best == number - 1){
        cout << "(手气最佳)" << endl;
    }
    else {
        
        cout << endl;
    }
    return 0;
}
```

# 4、类中特殊成员变量的初始化问题：

常量变量：必须通过构造函数参数列表进行初始化。
引用变量：必须通过构造函数参数列表进行初始化。
普通静态变量：要在类外通过"::"初始化。
静态整型常量：可以直接在定义的时候初始化。
静态非整型常量：不能直接在定义的时候初始化。要在类外通过"::"初始化。

# 5、文件读写

```
#include <fstream> 
#include <string>
#include <iostream>
#include <streambuf> 

using namespace std; 

int WriteFile(string filepath,const string& Init)
{
    //定义文件输出流 
    ofstream outfile; 

    //根据参数路径创建要输出的文件 
    outfile.open(filepath, ios::out | ios::trunc);    
    if(!outfile)
        return 1;
    outfile << Init << endl;

    outfile.close(); 
    return 0;
}

string Read_Str(string filepath)
{
    ifstream infile;
    infile.open(filepath);
    //打开失败，路径不正确
    if(!infile)
        cout << "Open File Fail!" << endl;
    //读取文本内容到字符串
    string readStr((std::istreambuf_iterator<char>(infile)),  std::istreambuf_iterator<char>()); 
    
    return readStr;
}

int main()
{
    //执行完成在mian.cpp同一目录下生产C.txt文件
    WriteFile("./C.txt","Hello , world!");
    cout << Read_Str("./C.txt") << endl;

    getchar();
    return 0;
}
```

```
#include "iostream"
#include "fstream"
using namespace std;

//向文件内部写入数据，并将数据读出
void file_wr(void)
{
    char data[100];
    //向文件写入数据
    ofstream outfile;
    outfile.open("test.txt");
    cout << "Write to the file" << endl;
    cout << "Enter your name:" << endl;
    cin.getline(data, 100);
    outfile << data << endl;
    cout << "Enter your age:" << endl;
    cin >> data;
    cin.ignore();
    outfile << data << endl;
    outfile.close();
    //从文件读取数据
    ifstream infile;
    infile.open("test.txt");
    cout << "Read from the file" << endl;
    infile >> data;
    cout << data << endl;
    infile >> data;
    cout << data << endl;
    infile.close();
}


//将数据从一文件复制到另一文件中
void file_copy(void)
{
    char data[100];
    ifstream infile;
    ofstream outfile;
    infile.open("test.txt");
    outfile.open("test_1.txt");
    cout << "copy from test.txt to test_1.txt" << endl;
    while (!infile.eof())
    {
        infile >> data;
        cout << data << endl;
        outfile << data << endl;
    }
    infile.close();
    outfile.close();
}

//测试上述读写文件，与文件数据复制
int _tmain(int argc, _TCHAR* argv[])
{
    file_wr();
    file_copy();
    return 0;
}
```

```
关于 笔记1 中的复制操作。由于 eof 指示是在读取文件到结尾的时候，才会改变有效的状态。但是，再下一次没有读到数据的时候，eof 才会改变；

但是如果此时还是用 eof 标志位来判断文件是否读到了 end（下图while循环），就会导致此时的 infile>>data 语句还会流入数据（文件的最后一行数据），导致复制的文件中，会多出一行。

while (!infile.eof())
{
    infile >> data;
    cout << data << endl;
    outfile << data << endl;
}
按照 笔记1 中的输入，最后文件 copy 的结果应该是：

copy from test.txt to test_1.txt
John
20
20
为消除多与的读取、复制，我们只需要在还能读取文件数据的时候，才将数据复制到新文件中即可，即代码可以改成：

while (infile >> data)
{
    cout << data << endl;
    outfile << data << endl;
}
```

