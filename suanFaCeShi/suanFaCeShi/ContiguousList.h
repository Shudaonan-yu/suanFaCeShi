#pragma once
class ContiguousList
{
public:

#pragma region 顺序表操作
//静态分配
//#define MaxSize 10
//	typedef struct {
//		int data[MaxSize];
//		int length;
//	}SqList;
//void static InitList(SqList& L);
	//动态分配
#define InitSize 10
	typedef struct {
		int* data; //指示动态分配数组的指针
		int MaxSzie;//顺序表的最大容量
		int length;//顺序表的当期长度
	}SqList;
	// 初始化一个顺序表,用动态数组
	void static InitList_dymantic(SqList& L);

	//增加动态数组的长度
	void static IncreaseSize(SqList& L, int len);
		


#pragma endregion

	
	ContiguousList();
	~ContiguousList();
private:

};

