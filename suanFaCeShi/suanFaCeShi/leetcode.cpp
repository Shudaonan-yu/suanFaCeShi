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
#pragma region �����㷨����389��

//�ⷨ1������������㣬������a^b^c^a^b^c^d =d�����¡�
//ע��Ҫ���ַ���ת��Ϊ�ַ�*/
char leetcode::findTheDifference(string s, string t) {
	int ret = 0;
	for (char ch : s) {  //д�� char &ch :s ���Ա�ԭд����Լ�ڴ棬ԭ����&chֱ����ԭ�ַ���s�Ͻ��б�����ԭд��Ҫ���Ƴ���s�ٽ��б�������
		ret ^= ch;
	}
	for (char ch : t) {
		ret ^= ch;
	}
	return ret;
}

////�ⷨ2�����ý�ÿ���ַ���ascii����ͣ�Ȼ�����ַ���1��
////�ͼ�ȥ�ַ���2�ĺͼȴ����˱���ӵ��ַ�

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

#pragma region ����ƥ������
//����һ��ֻ���� '('��')'��'{'��'}'��'['��']' ���ַ������ж��ַ����Ƿ���Ч   ��Ч�ַ��������㣺�����ű�������ͬ���͵������űպϡ������ű�������ȷ��˳��պϡ�ע����ַ����ɱ���Ϊ����Ч�ַ���
bool leetcode::isValid(string s) {
	if (s.length() % 2 != 0)// ������s.length() & 1 == 1 ���жϳ����Ƿ�Ϊż����λ����ִ��Ч�ʸ�
		return false;
	if (s.length() == 0)
		return true;
	stack<char> mystack;
	int i = 0;
	mystack.push('#');//�ڱ�Ԫ�أ������ݽṹ�������жϱ߽磬���ٱ����Ŀ���
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
		// ��������� �Ǳ����ַ���ƥ��Ĺ����У�ջ�Ѿ�Ϊ���ˣ�û��ƥ����ַ��ˣ�˵��������û���ҵ���Ӧ�������� return false
		// �ڶ������ �����ַ���ƥ��Ĺ����У�����ջ��û������Ҫƥ����ַ�������return false
		else if (st.empty() || st.top() != s[i]) return false;
		else st.pop(); // st.top() == s[i]
	}
	// ��һ����� ��ʱ�����Ѿ����������ַ���������ջ��Ϊ�գ�˵������Ӧ��������û����������ƥ�䣬����return false�������return true
	return st.empty();
}

#pragma endregion

#pragma region �ֽ�����3.���ظ��ַ�����Ӵ�
int leetcode::lengthOfLongestSubstring(string s)
{
	//��ϣ���ϣ���¼ÿ���ַ��Ƿ���ֹ�
	unordered_set<char>occ;
	int n = s.size();
	//��ָ�룬��ʼֵΪ-1���൱�����ǻ����ַ��������߽磬��û�п�ʼ�ƶ�
	int rk = -1, ans = 0;
	//ö����ָ���λ�ã���ʼֵ���Ա���Ϊ-1
	for (int i = 0; i < n; ++i) {
		if (i!= 0) {
			//��ָ�������ƶ�һ���Ƴ�һ���ַ�
			occ.erase(s[i - 1]);
		}
		while (rk + 1 < n && !occ.count(s[rk + 1])){
			//�����ƶ���ָ��
			occ.insert(s[rk + 1]);
			++rk;
		}
		//��i����rk���ַ���һ�����������ظ��ַ��Ӵ�
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
		// �������ظ�ֵ��˵����ָ����Ҫ��ת����ת��λ���Ǹ��ظ�ֵ���±�+1
		// �����ַ���abcdecf���������ڶ���c�������bcde����һ����ʼ�����ȶ��޷�����a��ֻ�д�decf��ʼ���������һ�ֲ���
		// ֵ��ע����ǣ�����������ظ�ֵ���±����ָ�뻹С���������Ӧ����ת����Ϊ��ָ����ߵ�Ԫ�ز��ٴ����ڣ�����abba
		if (map.count(s[end]) && map.find(s[end])->second >= start) {
			start = map.find(s[end])->second + 1;
			map[s[end]] = end;//c++ ֻ����������valUe��ֵ
		}
		map.insert(pair<char, int>(s[end], end));// �����ز��ظ�����Ҫ���£���Ԫ��������±�
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

#pragma region 76.��С�����Ӵ�
string leetcode::minWindow(string s, string t)
{
	if (s.size() == 0 || t.size() == 0)
		return "�������ַ�";
	unordered_map<char, int>lookUp, minSubstring;

	return string();
}
#pragma endregion




#pragma region 쳲���������
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

//���鴥��
vector<int> leetcode::getTriggerTime(vector<vector<int> >& increase, vector<vector<int> >& requirements)
{
	vector<vector<int> > s(increase.size() + 1, vector<int>(3, 0));//vector<vector<int> > ��ÿ�һ�񣬷����еı������ᱨ��
	for (int i = 0; i < increase.size(); i++) {
		for (int j = 0; j < 3; j++) {
			s[i + 1][j] = s[i][j] + increase[i][j];
		}
	}
	vector<int> ans;
	for (auto v : requirements) {
		int l = 0, r = increase.size();  // ���ֲ���
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



//��������λ��
int leetcode::searchInsert(vector<int>& nums, int target)
{
	if (nums.empty()) return NULL;
	int l = 0, r = nums.size() - 1;//����ָ��
	while (l<r)
	{
		int mid = l+((r-l)>>1);//�����е�ȡ��
		if (nums[mid] >= target)
			r = mid;
		else
			l = mid + 1;
	}
	if (nums[l] < target) return l + 1;
	else return l;//ѭ������ʱ l = r,���Է����ĸ���һ��
}

//81. ������ת�������� II
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

//74.������ά����
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

//������ˢ���ؼ��ⷨ
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

//240.������ά����II
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

//	53. Ѱ����ת���������е���Сֵ
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
		int mid = l + ((r+1 - l) >> 1); /* �ȼ�һ�ٳ���mid�������ұߵ�right */
		if (nums[mid] > nums[l]) {
			l = mid;
		}
		else
			if (nums[mid] < nums[l]) {
				r = mid - 1;
			}
	}
	return nums[(r + 1) % nums.size()];/* ���ֵ�����ƶ�һλ������Сֵ�ˣ���Ҫ�������ֵ�����ұߵ����������һλ������鳤��ȡ�ࣩ */
}

int leetcode::finMin4(vector<int>& nums)
{
	int l = 0, r = nums.size() - 1;
	while (l <= r) { // ѭ��������ѡΪ����ұ�����left <= right
		int mid = l + ((r - l) >> 1);
		if (nums[mid] >= nums[r]) {// ע���ǵ���ֵ���ڵ�����ֵʱ��
			l = mid + 1; // ����߽��ƶ�����ֵ���ұ�
		}
		else { // ����ֵС����ֵʱ
			r = mid;      // ���ұ߽��ƶ�����ֵ��
		}
	}

	return nums[r];      // ��Сֵ����nums[right]
}



int leetcode::findMin3(vector<int>& nums) {
	int left = 0;
	int right = nums.size() - 1;                /* ����ұ����䣬������ҿ������򲻷����ж���ֵ */
	while (left < right) {                      /* ѭ������ʽ�����left == right����ѭ������ */
		int mid = left + (right - left) / 2;    /* �ذ����mid������left */
		if (nums[mid] > nums[right]) {          /* ��ֵ > ��ֵ����Сֵ���Ұ�ߣ�������߽� */
			left = mid + 1;                     /* ��Ϊ��ֵ > ��ֵ����ֵ�϶�������Сֵ����߽���Կ��mid */
		}
		else if (nums[mid] < nums[right]) {   /* ��ȷ��ֵ < ��ֵ����Сֵ�����ߣ������ұ߽� */
			right = mid;                        /* ��Ϊ��ֵ < ��ֵ����ֵҲ��������Сֵ���ұ߽�ֻ��ȡ��mid�� */
		}
	}
	return nums[left];    /* ѭ��������left == right����Сֵ���nums[left]��nums[right]���� */
}
#pragma endregion

//1018.�ɱ�5�����Ķ�������
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

#pragma region λ����ʵ�ּӼ��˳�
int leetcode::add(int a, int b)
{
	if (b == 0) //�ݹ��������������Ҽ���Ϊ0���������н�λ�ˣ��������
		return a;
	int s = a^b;
	int c = (a&b) << 1; //��λ����1λ���ﵽ��λ��Ŀ�ġ�
	return add(s, c); //�ٰ�'��'��'��λ'��ӡ��ݹ�ʵ�֡�
}
int leetcode::subTraction(int a, int b)
{
	return add(a,negative(b));
}


//��ֱ�ۣ�������ѭ���ӷ�����˷���a*b�����ǰ�a�ۼ�b�Ρ�ʱ�临�Ӷ�ΪO(N)��
int leetcode::multiply(int a, int b)
{
	bool flag = true;
	if (getSign(a) == getSign(b)) {//���ķ����ж�
		flag = false;
	}
	a = bePositive(a);//�Ȱѳ����ͱ�������Ϊ����
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
		double d = (double)i /(double) K; //�����������ȡ��Ҳ����д�ɣ�i -1)/K + 1;
		sum += ceil(d);
	}
	
	return (sum > H) ? false: true;
}

int leetcode::minEatingSpeed(vector<int>& piles, int H)
{
	int l = 1;
	auto r = max_element(piles.begin(), piles.end());//��ʱr���ָ����ָ�������е����Ԫ��
	int rr = (int)*r;
	while (l < rr) {//���ص��ǵ�����Ԫ�أ�Ҫ��*ȡ�����д��ֵ
		int mid = l + ((rr - l) >> 1);
		if (!possible(piles, H, mid)) {
			l = mid + 1;
		}
		else
			rr = mid;//���д��*r = mid,�����е����ֵҲ�ᱻ�޸�
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


