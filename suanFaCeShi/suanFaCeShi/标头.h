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
	typedef A::a_type type1; //��ȷ������ʹ��A��a_type��
	typedef typename param::a_type type2;
	//����Ҫ��param ǰ�����typename���������˵�����ģ�����a_typeָ����һ������

	C() {
		c = value::a_type;//ģ���βεľ�̬��Ա����
	}

	int c;
};


int main() {

	C<A, B> c;

	return 0;
}
#pragma once
