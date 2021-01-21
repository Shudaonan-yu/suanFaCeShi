#pragma once
#include <string>
#include <vector>
#include<stack>
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

#pragma region 04 ��ά�����еĲ���
	bool findNumberIn2DArray(vector<vector<int>>& matrix, int target);
#pragma endregion

#pragma region 05.�滻�ո�
/*�ⷨ1������ÿ���滻�� 1 ���ַ���� 3 ���ַ���ʹ���ַ�����ɷ���ؽ����滻��
�����ַ�����س���Ϊ s �ĳ��ȵ� 3 ���������ɱ�֤�ַ�����������������滻����ַ���

1.��� s �ĳ��� length
2.�����ַ����� array���䳤��Ϊ length * 3
3.��ʼ�� size Ϊ 0��size ��ʾ�滻����ַ����ĳ���
4.�����ұ����ַ��� s
5.��� s �ĵ�ǰ�ַ� c
6.����ַ� c �ǿո����� array[size] = '%'��array[size + 1] = '2'��array[size + 2] = '0'������ size ��ֵ�� 3
7.����ַ� c ���ǿո����� array[size] = c������ size ��ֵ�� 1
8.��������֮��size ��ֵ�����滻����ַ����ĳ��ȣ��� array ��ǰ size ���ַ��������ַ��������������ַ���
*/
	string replaceSpace(string s);



/*�ⷨ2��ʹ��˫ָ��
* 
*/
	string replaceSpace2(string s);
#pragma endregion

#pragma region 06.��β��ͷ��ӡ����
	vector<int> ans;

	//�Լ��Ľⷨ���÷�ת����ķ�ʽ���
	vector<int> reversePrint(ListNode* head);

	/*�ݹ�ⷨ
	* ���ƽ׶Σ� ÿ�δ��� head.next ���� head == null�����߹�����β���ڵ㣩Ϊ�ݹ���ֹ��������ʱֱ�ӷ��ء�
���ݽ׶Σ� ������ʱ������ǰ�ڵ�ֵ�����б���tmp.add(head.val)��
���գ����б� tmp ת��Ϊ���� res �������ؼ��ɡ�

���ߣ�jyd
���ӣ�https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/mian-shi-ti-06-cong-wei-dao-tou-da-yin-lian-biao-d/
��Դ�����ۣ�LeetCode��
����Ȩ���������С���ҵת������ϵ���߻����Ȩ������ҵת����ע��������
	*/
	vector<int> reversePrintByRecursion(ListNode* head);


	/* �ⷨ3������ջ��
	* 
	*/
	vector<int> reversePrintByStack(ListNode* head);
#pragma endregion



#pragma region �ؽ�������
	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
		
	};
	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);

	TreeNode* recursionBuild(vector<int>::iterator preBegin, vector<int>::iterator preEnnd, vector<int>::iterator inBegin, vector<int>::iterator inEnd);
#pragma endregion


#pragma region 09.������ջʵ�ֶ���
	class CQueue {
		stack<int> stack1, stack2;
	public:
		CQueue() {
			//��ʼ���������ǵ�����ջ��Ϊ��ʱ����������ջ��Ϊ��
			while (!stack1.empty()) {
				stack1.pop();
			}
			while (!stack2.empty()) {
				stack2.pop();
			}

		}

		void appendTail(int value) {
			stack1.push(value);
		}

		int deleteHead() {
			if (stack2.empty()) {
				while (!stack1.empty()) {
					stack2.push(stack1.top());
					stack1.pop();
				}
			}
			if (stack2.empty()) {
				return -1;
			}
			else {
				int deleteItem = stack2.top();
				stack2.pop();
				return deleteItem;
			}
		}
	};
#pragma endregion


#pragma region 11 ��ת�������С����
	int minArray(vector<int>& numbers);
#pragma endregion


	
private:




};




#endif