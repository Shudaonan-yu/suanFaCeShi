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

#pragma region 04 二维数组中的查找
	bool findNumberIn2DArray(vector<vector<int>>& matrix, int target);
#pragma endregion

#pragma region 05.替换空格
/*解法1：由于每次替换从 1 个字符变成 3 个字符，使用字符数组可方便地进行替换。
建立字符数组地长度为 s 的长度的 3 倍，这样可保证字符数组可以容纳所有替换后的字符。

1.获得 s 的长度 length
2.创建字符数组 array，其长度为 length * 3
3.初始化 size 为 0，size 表示替换后的字符串的长度
4.从左到右遍历字符串 s
5.获得 s 的当前字符 c
6.如果字符 c 是空格，则令 array[size] = '%'，array[size + 1] = '2'，array[size + 2] = '0'，并将 size 的值加 3
7.如果字符 c 不是空格，则令 array[size] = c，并将 size 的值加 1
8.遍历结束之后，size 的值等于替换后的字符串的长度，从 array 的前 size 个字符创建新字符串，并返回新字符串
*/
	string replaceSpace(string s);



/*解法2：使用双指针
* 
*/
	string replaceSpace2(string s);
#pragma endregion

#pragma region 06.从尾到头打印链表
	vector<int> ans;

	//自己的解法，用反转链表的方式输出
	vector<int> reversePrint(ListNode* head);

	/*递归解法
	* 递推阶段： 每次传入 head.next ，以 head == null（即走过链表尾部节点）为递归终止条件，此时直接返回。
回溯阶段： 层层回溯时，将当前节点值加入列表，即tmp.add(head.val)。
最终，将列表 tmp 转化为数组 res ，并返回即可。

作者：jyd
链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/mian-shi-ti-06-cong-wei-dao-tou-da-yin-lian-biao-d/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
	*/
	vector<int> reversePrintByRecursion(ListNode* head);


	/* 解法3：辅助栈法
	* 
	*/
	vector<int> reversePrintByStack(ListNode* head);
#pragma endregion



#pragma region 重建二叉树
	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
		
	};
	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);

	TreeNode* recursionBuild(vector<int>::iterator preBegin, vector<int>::iterator preEnnd, vector<int>::iterator inBegin, vector<int>::iterator inEnd);
#pragma endregion


#pragma region 09.用两个栈实现队列
	class CQueue {
		stack<int> stack1, stack2;
	public:
		CQueue() {
			//初始化方法就是当两个栈不为空时，将这两个栈置为空
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


#pragma region 11 旋转数组的最小数字
	int minArray(vector<int>& numbers);
#pragma endregion


	
private:




};




#endif