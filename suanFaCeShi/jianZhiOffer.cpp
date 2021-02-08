#include "jianZhiOffer.h"
#include<string>
#include <stack>
#include<vector>
#include<iostream>
#include<queue>
#include<unordered_set>
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
bool jianZhiOffer::findNumberIn2DArray(vector<vector<int>>& matrix, int target)
{
	if (matrix.empty()) return false;
	int m = matrix[0].size();
	int line = matrix.size() - 1, row = 0;
	while (line >= 0 && row < m) {//ע���Ǵ��ڵ���0
		if (matrix[line][row] == target) {
			return true;
		}
		else if (matrix[line][row] > target)
			line -= 1;
		else
			row += 1;
	}
	return false;
}
string jianZhiOffer::replaceSpace(string s)
{
	if (s.length() < 0 || s.length() > 10000) return "@@";
	string arry;
	for (char& c : s) {
		if (c == ' ') {
			arry.push_back('%');
			arry.push_back('2');
			arry.push_back('0');
		}
		else
			arry.push_back(c);
	}
	return arry;
}
string jianZhiOffer::replaceSpace2(string s)
{
	int count = 0; // ͳ�ƿո�ĸ���
	int sOldSize = s.size();
	for (int i = 0; i < sOldSize; i++) {
		if (s[i] == ' ') {
			count++;
		}
	}
	//�����ַ���s�Ĵ�С��Ҳ����ÿ���ո��滻�ɡ�%20��֮��Ĵ�С
	s.resize(sOldSize + (count << 1));//֮ǰд��sOldSize + count << 1������ԭ����<<�����ȼ���� ����Ҫ�����ţ��м��м�
	int sNewSize = s.size();
	//�Ӻ���ǰ���ո��滻Ϊ��%20��
	for (int i = sNewSize - 1, j = sOldSize - 1; j < i; --i, --j) {
		if (s[j] != ' ') {
			s[i] = s[j];
		}
		else {
			s[i] = 0;
			s[i - 1] = '2';
			s[i - 2] = '%';
			i -= 2;
		}
	}
	return s;
}
vector<int> jianZhiOffer::reversePrint(ListNode* head)
{
	ListNode* cur = head;
	ListNode* next = NULL;
	ListNode* pre = NULL;
	while (cur){
		next = cur->next;
		cur->next = pre;
		pre = cur;
		cur = next;
	}
	ListNode* p = pre;
	while (p) {
		ans.emplace_back(p->val);
		p = p->next;
	}
	return ans;
}
vector<int> jianZhiOffer::reversePrintByRecursion(ListNode* head)
{
	if (head == nullptr) {
		return {};
	}
	reversePrintByRecursion(head->next);
	ans.emplace_back(head->val);
	return ans;
}
vector<int> jianZhiOffer::reversePrintByStack(ListNode* head)
{
	stack<int> s;

	//��ջ
	while (head) {
		s.push(head->val);
		head = head->next;
	}
	//��ջ
	while (!s.empty()) {
		ans.emplace_back(s.top());
		s.pop();
	}
	return ans;
}
jianZhiOffer::TreeNode* jianZhiOffer::buildTree(vector<int>& preorder, vector<int>& inorder)
{
	if (preorder.empty())
		return NULL;
	return recursionBuild(preorder.begin(), preorder.end(), inorder.begin(), inorder.end());//����ԭ���������������½���������ʡ�ռ�
}

jianZhiOffer::TreeNode* jianZhiOffer::recursionBuild(vector<int>::iterator preBegin, vector<int>::iterator preEnnd, vector<int>::iterator inBegin, vector<int>::iterator inEnd) {
	if (inEnd == inBegin) return NULL;
	jianZhiOffer::TreeNode* cur = new jianZhiOffer::TreeNode(*preBegin);//�½�ָ��ָ��������������ĵ�һ�����
	auto root = find(inBegin, inEnd, *preBegin);//������������������ҵ���Ӧ�ģ���Ϊ�����ĸ���㣬Ҳ�����������������ķֽ��
	//���������root - ���������bigin ��Ϊ��ȷ�����������������ж��ٽ��,vector��iterator��end��ָ�����һ��Ԫ�ص���һλ������Ҳ��������˵��������ǰ�պ�
	cur->left = recursionBuild(preBegin + 1, preBegin + 1 + (root - inBegin), inBegin, root);//���������������ȷ���ĸ���㽫�������һ��Ϊ������������������ұ���������
	cur->right = recursionBuild(preBegin + 1 + (root - inBegin),preEnnd,root + 1,inEnd);
	return cur;
}
#pragma endregion



int jianZhiOffer::minArray(vector<int>& numbers)
{
	if (numbers.empty()) return -1;
	int l = 0, r = numbers.size() - 1;
	while (l < r) {
		int mid = l + ((r - l) >> 1);
		while (mid < r && numbers[mid] == numbers[r]) {
			r = r - 1;
		}
		if (numbers[mid] > numbers[r]) {
			l = mid + 1;
		}
		else {
			r = mid;
		}
	}
	return numbers[l];
}

jianZhiOffer::ListNode* jianZhiOffer::deleteNode(ListNode* head, int val)
{
	ListNode* dummyHead = new ListNode(0);
	dummyHead->next = head;
	ListNode* cur = dummyHead;
	while (cur->next != NULL) {
		if (cur->next->val == val) {
			ListNode* temp = cur->next;
			cur->next = cur->next->next;
			//delete temp;
		}
		else {
			cur = cur->next;
		}
	}
	return dummyHead->next;
}

int jianZhiOffer::numWays(int n)
{
	int a = 1, b = 1, sum;
	for (int i = 0; i < n; i++) {
		sum = (a + b) % 1000000007;
		a = b;
		b = sum;
	}
	return a;
}

jianZhiOffer::TreeNode* jianZhiOffer::mirrorTree(TreeNode* root)
{
	
	if (root == NULL) return NULL;//�ݹ���ֹ���������ʵ���Ҷ�ӽڵ�Ĳ����ڵ���һ�㣬��������ӵݹ麯������������ʼ���Ϸ���
	swap(root->left, root->right);//swap�������Խ���NULLֵ
	mirrorTree(root->left);
	mirrorTree(root->right);
	return root; //���Ϸ��ص�ǰ�ڵ�,��Ϊÿ���ڵ㶼�Ѿ����Ƚ������ˣ���󷵻��������ĸ��ڵ�
}

jianZhiOffer::TreeNode* jianZhiOffer::mirrorTree2(TreeNode* root)
{
	stack<TreeNode*>s;
	s.push(root);
	while (!s.empty()) {
		TreeNode* cur = s.top();
		s.pop();
		if (cur == NULL) {
			continue;//��ǰ�ڵ�Ϊ��ʱ������һ��ѭ��
		}
		swap(cur->left, cur->right);
		s.push(cur->left);
		s.push(cur -> right);
	}
	return root;//return root���ڷ���������
}

jianZhiOffer::TreeNode* jianZhiOffer::mirrorTree3(TreeNode* root)
{
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty()) {
		TreeNode* cur = q.front();
		q.pop();
		if (cur == NULL) {
			continue;
		}
		swap(cur->left, cur->right);
		q.push(cur->left);
		q.push(cur->right);
	}
	return root;
}

int jianZhiOffer::findRepeatNumber(vector<int>& nums)
{
	unordered_set<int>list;
	int repeat = -1;
	for (int i : nums) {
		if (list.find(i) != list.end()) {
			repeat = i;
			break;
		}
		else {
			list.insert(i);
		}
	}
	return repeat;
}




