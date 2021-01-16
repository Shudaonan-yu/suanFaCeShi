class A {
public:
	typedef int a_type;
	int a;
};

class B
{
	B() {}

public:
	static int a_type;

};

int B::a_type = 1;

template <typename param, typename value>
class C {
public:
	typedef A::a_type type1; //正确，正常使用A的a_type类
	typedef typename param::a_type type2;
	//必须要在param 前面加上typename，向编译器说明这个模块类的a_type指的是一种类型

	C() {
		c = value::a_type;//模板形参的静态成员变量
	}

	int c;
};


int main() {

	C<A, B> c;

	return 0;
}
#pragma once
