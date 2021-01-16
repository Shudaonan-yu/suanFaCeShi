#include "jianZhiOffer.h"
#include<string>
#include <stack>
#include<vector>
#include<iostream>
using namespace std;

jianZhiOffer::jianZhiOffer() {};
jianZhiOffer::~jianZhiOffer() {};


#pragma region 10-ii ������̨������
//int jianZhiOffer::numWays(int n) {
//
//  }

#pragma endregion




#pragma region ����08.06 ��ŵ������
void jianZhiOffer::move(int n, char from, char to) {
	printf("��%d��������%d�����Ӵ�%c����------->%c\n", i++, n, from, to);
}

void jianZhiOffer::hanNuo(int n, char star_pos, char trans_pos, char end_pos) {
	if (n == 1) move(n, star_pos, end_pos);
	else {
		hanNuo(n - 1, star_pos, end_pos, trans_pos);
		move(n, star_pos, end_pos);
		hanNuo(n - 1, trans_pos, star_pos, end_pos);
	}
}

//��ջ�ķ�ʽ�����ƶ�
void jianZhiOffer::move2(int n, vector<int>& from, vector<int>& to)
{
		to.push_back(from.back());
		from.pop_back();
}
void jianZhiOffer::hanNuo_Stack(int n, vector<int>& strat, vector<int>& trans, vector<int>& end)
{
	if (n == 1) move2(n, strat, end);
	else {
		hanNuo_Stack(n - 1, strat, end, trans);
		move2(n, strat, end);
		hanNuo_Stack(n - 1, trans, strat, end);
	}
}
#pragma endregion

 #pragma region �������
jianZhiOffer::ListNode * jianZhiOffer::addTwoNumbers(ListNode * l1, ListNode * l2)
{
	if (l1 == NULL && l2 == NULL) return NULL;
	else if (l1 == NULL)
	{
		return l2;
	}
	else if (l2 == NULL) {
		return l1;
	}
	int jinwei = 0;//�洢��λ��
	int sum = 0;
	ListNode* head = new ListNode(-1),*p1 = l1,*p2 = l2, *p = head;
	while (p1 != NULL || p2!= NULL||jinwei)
	{
		sum = 0;
		if (p1) {
			sum += (p1->val);
			p1 = p1->next;
		}
		if (p2) {
			sum += (p2->val);
			p2 = p2->next;
		}
		sum += jinwei;
		ListNode* s = new ListNode(sum%10);
		jinwei = sum / 10;
		p->next = s;
		p = s;	
	}
	
	return head->next;
}
#pragma endregion



