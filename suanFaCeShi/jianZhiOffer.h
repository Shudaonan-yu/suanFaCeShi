#pragma once
#include <string>
#include <vector>
using namespace std;
#ifndef jianZhiOffe_H
#define jianZhiOffe_H
class jianZhiOffer {
public:
	jianZhiOffer();
	~jianZhiOffer();

	//剑指 Offer 10- II. 青蛙跳台阶问题
	int numWays(int n);

	//面试题08.06 汉若塔问题
	int i = 0;
	void move(int n, char from, char to);
	void hanNuo(int n, char start_pos, char trans_pos, char end_pos);

	//用栈的方式移动表示数据移动
	void move2(int n, vector<int>& from, vector<int>& to);
	void hanNuo_Stack(int n, vector<int>& strat, vector<int>& trans, vector<int>& end);


#pragma region 面试题02.05 链表求和
/*描述：给定两个用链表表示的整数，每个节点包含一个数位。

这些数位是反向存放的，也就是个位排在链表首部。

编写函数对这两个整数求和，并用链表形式返回结果。


示例： 输入：(7 -> 1 -> 6) + (5 -> 9 -> 2)，即617 + 295
输出：2 -> 1 -> 9，即912
*/
	typedef struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}	
}LisNode;

	//比较直白的 模拟手工求和的思路，新建一个链表，空间复杂度高
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

#pragma endregion





	
private:




};




#endif