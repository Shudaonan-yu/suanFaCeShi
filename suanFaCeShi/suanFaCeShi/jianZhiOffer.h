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

	//��ָ Offer 10- II. ������̨������
	int numWays(int n);

	//������08.06 ����������
	int i = 0;
	void move(int n, char from, char to);
	void hanNuo(int n, char start_pos, char trans_pos, char end_pos);

	//��ջ�ķ�ʽ�ƶ���ʾ�����ƶ�
	void move2(int n, vector<int>& from, vector<int>& to);
	void hanNuo_Stack(int n, vector<int>& strat, vector<int>& trans, vector<int>& end);


#pragma region ������02.05 �������
/*���������������������ʾ��������ÿ���ڵ����һ����λ��

��Щ��λ�Ƿ����ŵģ�Ҳ���Ǹ�λ���������ײ���

��д������������������ͣ�����������ʽ���ؽ����


ʾ���� ���룺(7 -> 1 -> 6) + (5 -> 9 -> 2)����617 + 295
�����2 -> 1 -> 9����912
*/
	typedef struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}	
}LisNode;

	//�Ƚ�ֱ�׵� ģ���ֹ���͵�˼·���½�һ�������ռ临�Ӷȸ�
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

#pragma endregion





	
private:




};




#endif