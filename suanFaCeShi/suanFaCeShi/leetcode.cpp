#include "leetcode.h"
#include<string>
#include <stack>
#include<unordered_set>
#include<math.h>
#include<algorithm>
#include<unordered_map>
using namespace std;


leetcode::leetcode()
{

}
leetcode::~leetcode()
{

}
#pragma region 力扣算法题库第389题

//解法1，利用异或运算，利用了a^b^c^a^b^c^d =d的兴致。
//注意要讲字符串转化为字符*/
char leetcode::findTheDifference(string s, string t) {
	int ret = 0;
	for (char ch : s) {  //写成 char &ch :s 可以别原写法节约内存，原因是&ch直接在原字符串s上进行遍历，原写法要复制出个s再进行遍历操作
		ret ^= ch;
	}
	for (char ch : t) {
		ret ^= ch;
	}
	return ret;
}

////解法2，利用将每个字符的ascii码求和，然后用字符串1的
////和减去字符串2的和既代表了被添加的字符

char leetcode::findTheDifference2(string s, string t) {
	int ret1 = 0;
	int ret2 = 0;
	for (char ch : s) {
		ret1 += ch;
	}
	for (char ch : t) {
		ret2 += ch;
	}
	return abs( ret2 - ret1);
}

#pragma endregion

#pragma region 括号匹配问题
//给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效   有效字符串需满足：左括号必须用相同类型的右括号闭合。左括号必须以正确的顺序闭合。注意空字符串可被认为是有效字符串
bool leetcode::isValid(string s) {
	if (s.length() % 2 != 0)// 可以用s.length() & 1 == 1 来判断长度是否为偶数，位运算执行效率高
		return false;
	if (s.length() == 0)
		return true;
	stack<char> mystack;
	int i = 0;
	mystack.push('#');//哨兵元素，在数据结构中用于判断边界，减少遍历的开销
	while (i < s.size()) {
		if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
			mystack.push(s[i]);
			++i;
		}
		else if ((s[i] == ')' && mystack.top() == '(') || (s[i] == ']' && mystack.top() == '[') || (s[i] == '}' && mystack.top() == '{'))
		{
			 ++i;
			mystack.pop();
		}
		else
			return false;
	}

	if (mystack.top() == '#')
		return true;
		return false;
}

bool leetcode::isVaild2(string s) {
	stack<int> st;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '(') st.push(')');
		else if (s[i] == '{') st.push('}');
		else if (s[i] == '[') st.push(']');
		// 第三种情况 是遍历字符串匹配的过程中，栈已经为空了，没有匹配的字符了，说明右括号没有找到对应的左括号 return false
		// 第二种情况 遍历字符串匹配的过程中，发现栈里没有我们要匹配的字符。所以return false
		else if (st.empty() || st.top() != s[i]) return false;
		else st.pop(); // st.top() == s[i]
	}
	// 第一种情况 此时我们已经遍历完了字符串，但是栈不为空，说明有相应的左括号没有右括号来匹配，所以return false，否则就return true
	return st.empty();
}

#pragma endregion

#pragma region 字节跳动3.无重复字符的最长子串
int leetcode::lengthOfLongestSubstring(string s)
{
	//哈希集合，记录每个字符是否出现过
	unordered_set<char>occ;
	int n = s.size();
	//右指针，初始值为-1，相当于我们还在字符串的左侧边界，还没有开始移动
	int rk = -1, ans = 0;
	//枚举左指针的位置，初始值隐性表现为-1
	for (int i = 0; i < n; ++i) {
		if (i!= 0) {
			//左指针向右移动一格，移除一个字符
			occ.erase(s[i - 1]);
		}
		while (rk + 1 < n && !occ.count(s[rk + 1])){
			//不断移动右指针
			occ.insert(s[rk + 1]);
			++rk;
		}
		//第i到第rk个字符是一个极长的无重复字符子串
		ans = max(ans, rk - i + 1);
	}
	return ans;
}
int leetcode::lengthOfLongestSubstring_byMap(string s)
{
	if (s == "" || s.length() == 0) {
		return 0;
	}
	int ans = 0;
	int len = s.length();
	int start = 0;
	int end = 0;
	unordered_map<char, int>map;
	while (end < len) {
		ans = max(ans, end - start);
		// 当遇到重复值，说明左指针需要跳转，跳转的位置是该重复值的下标+1
		// 比如字符串abcdecf，到遇到第二个c，即便从bcde任意一个开始，长度都无法超过a，只有从decf开始计算才是新一轮查找
		// 值得注意的是，如果碰到了重复值的下标比左指针还小的情况，不应该跳转，因为左指针左边的元素不再窗口内，比如abba
		if (map.count(s[end]) && map.find(s[end])->second >= start) {
			start = map.find(s[end])->second + 1;
			map[s[end]] = end;//c++ 只能这样更新valUe的值
		}
		map.insert(pair<char, int>(s[end], end));// 无论重不重复都需要更新，该元素最近的下标
		end++;
	}
	return ans = max(ans, end - start);
}
int leetcode::lengthOfLongerstSubstring2(string s)
{
	if (s.size() == 0)
		return 0;
	unordered_set<char> lookUp;
	int maxStr = 0, left = 0;
	for (int i = 0; i < s.size(); i++) {
		while (lookUp.find(s[i]) != lookUp.end())
		{
			lookUp.erase(s[left]);
			left++;
		}
		maxStr = max(maxStr, i - left + 1);
		lookUp.insert(s[i]);
	}
	return maxStr;
}
#pragma endregion

#pragma region 76.最小覆盖子串
string leetcode::minWindow(string s, string t)
{
	if (s.size() == 0 || t.size() == 0)
		return "请输入字符";
	unordered_map<char, int>lookUp, minSubstring;

	return string();
}
#pragma endregion




#pragma region 斐波那契数列
int leetcode::fib_recursion(int n)
{
	if (n < 2) return n;

	return fib_recursion(n - 1) + fib_recursion(n - 2);
}

int leetcode::fib_gundong(int n)
{
	if (n < 2) return n;
	int p = 0, q = 0, r = 1;
	for (int i = 2; i <= n; ++i) {
		p = q;
		q = r;
		r = p + q;
	}
	return r;
}

int leetcode::fib_tongxiang(int n)
{
	double sqrt5 = sqrt(5);
	double fibN = pow((1 + sqrt5) / 2, n) - pow((1 - sqrt5) / 2, n);
	return round(fibN / sqrt(5));
}
#pragma endregion

//剧情触发
vector<int> leetcode::getTriggerTime(vector<vector<int> >& increase, vector<vector<int> >& requirements)
{
	vector<vector<int> > s(increase.size() + 1, vector<int>(3, 0));//vector<vector<int> > 最好空一格，否则有的编译器会报错
	for (int i = 0; i < increase.size(); i++) {
		for (int j = 0; j < 3; j++) {
			s[i + 1][j] = s[i][j] + increase[i][j];
		}
	}
	vector<int> ans;
	for (auto v : requirements) {
		int l = 0, r = increase.size();  // 二分查找
		while (l < r) {
			int m = (l + r) / 2;
			if (s[m][0] >= v[0] && s[m][1] >= v[1] && s[m][2] >= v[2])
				r = m;
			else
				l = m + 1;
		}
		if (s[l][0] >= v[0] && s[l][1] >= v[1] && s[l][2] >= v[2])
			ans.push_back(l);
		else
			ans.push_back(-1);
	}
	return ans;
}



//搜索插入位置
int leetcode::searchInsert(vector<int>& nums, int target)
{
	if (nums.empty()) return NULL;
	int l = 0, r = nums.size() - 1;//左右指针
	while (l<r)
	{
		int mid = l+((r-l)>>1);//最优中点取法
		if (nums[mid] >= target)
			r = mid;
		else
			l = mid + 1;
	}
	if (nums[l] < target) return l + 1;
	else return l;//循环结束时 l = r,所以返回哪个都一样
}

//81. 搜索旋转排序数组 II
bool leetcode::search(vector<int>& nums, int target)
{
	if (nums.empty()) return false;
	int l = 0, r = nums.size() - 1;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		if (nums[mid] == target) {
			return true;
		}
		while (l<mid && nums[l] == nums[mid])
		{
			l += 1;
		}
		if (nums[l] <= nums[mid]) {
			if (nums[l] <= target && target < nums[mid]) {
				r = mid - 1;
			}
			else
				l = mid + 1;

		}
		else {
			if (nums[mid] < target && target <= nums[r]) {
				l = mid + 1;
			}
			else
				r = mid - 1;
		}
	 
	}
	return false;
}

//74.搜索二维矩阵
bool leetcode::searchMatrix(vector<vector<int>>& matrix, int target)
{
	if(matrix.empty() || target < -10000 || target > 10000)
	return false;
	int m = matrix.size(),n = matrix[0].size();
	int l = 0, r = m * n - 1;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		int midH = mid / n, midV = mid % n;
		if (matrix[midH][midV] == target)
			return true;
		if (matrix[midH][midV] < target) {
			l = mid + 1;
		}
		else if(matrix[midH][midV] > target) {
			r = mid - 1;
		}
	}
	return false;
}

//西法的刷题秘籍解法
bool leetcode::searchMatrix2(vector<vector<int>>& matrix, int target)
{
	if (matrix.empty() || target < -10000 || target > 10000)
		return false;
	int m = matrix.size(), n = matrix[0].size();
	int x = m - 1, y = 0;
	while (x >= 0 && y < n) {
		if (matrix[x][y] < target) {
			y += 1;
		}
		else if (matrix[x][y] > target) {
			x -= 1;
		}
		else
			return true;
	}
	return false;
}

//240.搜索二维矩阵II
bool leetcode::searchMatrixEnhance(vector<vector<int>>& matrix, int target)
{
	int m = matrix.size(), n = matrix[0].size();
	int x = m - 1, y = 0;
	while (x >= 0 && y < n) {
		if (matrix[x][y] < target) {
			y += 1;
		}
		else if (matrix[x][y] > target) {
			x -= 1;
		}
		else
			return true;
	}
	return false;
}

//	53. 寻找旋转排序数组中的最小值
#pragma region
int leetcode::findMin(vector<int>& nums)
{
	if (nums.empty()) return -1;
	int l = 0, r = nums.size() - 1;
	while (l < r || nums[l] < nums[r]) {
		int mid = l + ((r - l) >> 1);
		if (nums[r] < nums[mid]) {
			l = mid +1 ;
		}
		else
		{
			r = mid ;
		}
	}
	return nums[l];
}

int leetcode::findMin2(vector<int>& nums)
{
	if (nums.empty()) return -1;
	int l = 0, r = nums.size() - 1;
	while (l <= r) {
		int mid = l + ((r - l) >> 1);
		if (nums[mid] > nums[mid + 1]) {
			return nums[mid + 1];
		}
		if (nums[mid - 1] > nums[mid]) {
			return nums[mid];
		}
		if (nums[mid] > nums[0]) {
			l = mid + 1;
		}
		else {
			r = mid - 1;
		}
	}
	return -1;
}

int leetcode::finMax(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int mid = l + ((r+1 - l) >> 1); /* 先加一再除，mid更靠近右边的right */
		if (nums[mid] > nums[l]) {
			l = mid;
		}
		else
			if (nums[mid] < nums[l]) {
				r = mid - 1;
			}
	}
	return nums[(r + 1) % nums.size()];/* 最大值向右移动一位就是最小值了（需要考虑最大值在最右边的情况，右移一位后对数组长度取余） */
}

int leetcode::finMin4(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l <= r) { // 循环的条件选为左闭右闭区间left <= right
		int mid = l + ((r - l) >> 1);
		if (nums[mid] >= nums[r]) {// 注意是当中值大于等于右值时，
			l = mid + 1; // 将左边界移动到中值的右边
		}
		else { // 当中值小于右值时
			r = mid;      // 将右边界移动到中值处
		}
	}

	return nums[r];      // 最小值返回nums[right]
}



int leetcode::findMin3(vector<int>& nums) {
	int left = 0;
	int right = nums.size() - 1;                /* 左闭右闭区间，如果用右开区间则不方便判断右值 */
	while (left < right) {                      /* 循环不变式，如果left == right，则循环结束 */
		int mid = left + (right - left) / 2;    /* 地板除，mid更靠近left */
		if (nums[mid] > nums[right]) {          /* 中值 > 右值，最小值在右半边，收缩左边界 */
			left = mid + 1;                     /* 因为中值 > 右值，中值肯定不是最小值，左边界可以跨过mid */
		}
		else if (nums[mid] < nums[right]) {   /* 明确中值 < 右值，最小值在左半边，收缩右边界 */
			right = mid;                        /* 因为中值 < 右值，中值也可能是最小值，右边界只能取到mid处 */
		}
	}
	return nums[left];    /* 循环结束，left == right，最小值输出nums[left]或nums[right]均可 */
}
#pragma endregion

//1018.可被5整除的二进制数
vector<bool> leetcode::prefixesDivBy5(vector<int>& A)
{
	vector<bool> ans;
	int subResult = 0;
	int length = A.size();
	for (int &i : A) {
		subResult = ((subResult << 1) + i) % 5;
		ans.emplace_back(subResult == 0);
	}
	return ans;
}

#pragma region 位运算实现加减乘除
int leetcode::add(int a, int b)
{
	if (b == 0) //递归结束条件：如果右加数为0，即不再有进位了，则结束。
		return a;
	int s = a^b;
	int c = (a&b) << 1; //进位左移1位，达到进位的目的。
	return add(s, c); //再把'和'和'进位'相加。递归实现。
}
int leetcode::subTraction(int a, int b)
{
	return add(a,negative(b));
}


//很直观，就是用循环加法替代乘法。a*b，就是把a累加b次。时间复杂度为O(N)。
int leetcode::multiply(int a, int b)
{
	bool flag = true;
	if (getSign(a) == getSign(b)) {//积的符号判定
		flag = false;
	}
	a = bePositive(a);//先把乘数和被乘数变为正数
	b = bePositive(b);
	int ans = 0;
	while (b) {
		ans = add(ans, a);
		b = subTraction(b, 1);
	}
	if (flag)
		ans = negative(ans);
	return ans;
}
#pragma endregion


int leetcode::findMinDuplicate(vector<int>& nums)
{
	if (nums.empty()) return -1;
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int mid = l + ((r - l) >> 1);
		while (mid < r && nums[mid] == nums[r]) {
			r = r - 1;
		}
		if (nums[mid] <= nums[r])
			r = mid;
		else {
			l = l + 1;
		}
	}
	return nums[l];
}

bool leetcode::possible(vector<int>& piles, int H, int K)
{
	int sum = 0;
	for (int &i : piles) {
		double d = (double)i /(double) K; //整数相除向上取整也可以写成（i -1)/K + 1;
		sum += ceil(d);
	}
	
	return (sum > H) ? false: true;
}

int leetcode::minEatingSpeed(vector<int>& piles, int H)
{
	int l = 1;
	auto r = max_element(piles.begin(), piles.end());//此时r这个指针是指向容器中的最大元素
	int rr = (int)*r;
	while (l < rr) {//返回的是迭代器元素，要用*取出其中存的值
		int mid = l + ((rr - l) >> 1);
		if (!possible(piles, H, mid)) {
			l = mid + 1;
		}
		else
			rr = mid;//如果写成*r = mid,容器中的最大值也会被修改
	}
	return l;
}
bool leetcode::possible2(vector<int>& weights, int D, int C) {
	int sumD = 0, sumW = 0;
	for (int &i : weights) {
		sumW += i;
		if (sumW > C) {
			sumD += 1;
			sumW = i;
		}
	}
	return (sumD + 1) <= D;
}
int leetcode::shipWithinDays(vector<int>& weights, int D)
{
	int l = (int)*max_element(weights.begin(),weights.end());
	int sum = 0;
	for (int &i : weights) {
		sum += i;
	}
	int r = sum;
	while (l < r) {
		int mid = l + ((r - l) >> 1);
		if (!possible2(weights, D, mid)) {
			l = mid + 1;
		}
		else
			r = mid;
			
	}
	return l;
}


