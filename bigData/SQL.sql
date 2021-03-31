-- 部门工资最高的员工
-- https://leetcode-cn.com/problems/department-highest-salary/solution/bu-men-gong-zi-zui-gao-de-yuan-gong-by-leetcode/

SELECT Department.name AS 'Department', Employee.name AS 'Employee', Salary
FROM Employee JOIN Department ON Employee.DepartmentId = Department.Id
WHERE
    (Employee.DepartmentId , Salary) IN
    (   SELECT
            DepartmentId, MAX(Salary)
        FROM
            Employee
        GROUP BY DepartmentId
	)


-- 部门工资前三高的员工
-- https://leetcode-cn.com/problems/department-top-three-salaries/solution/bu-men-gong-zi-qian-san-gao-de-yuan-gong-by-leetco/
SELECT d.Name AS 'Department', e1.Name AS 'Employee', e1.Salary
FROM 
    Employee e1 
        JOIN
    Department d on e1.DepartmentId = d.Id
WHERE
    3 > (
        SELECT 
            COUNT(DISTINCT e2.Salary)
        FROM 
            Employee e2
        WHERE 
            e2.Salary > e1.Salary AND e1.DepartmentId = e2.DepartmentId
    )


-- 连续出现的数字
-- https://leetcode-cn.com/problems/consecutive-numbers/solution/lian-xu-chu-xian-de-shu-zi-by-leetcode/
SELECT DISTINCT a.Num AS ConsecutiveNums
FROM 
    Logs a, Logs b, Logs c
WHERE
    a.ID = b.ID-1 AND b.ID = c.ID-1 AND a.Num = b.Num AND b.Num = c.Num


-- 178. 分数排名
-- https://leetcode-cn.com/problems/rank-scores/
SELECT a.Score AS Score,
    (select count(distinct  b.Score) FROM scores b WHERE b.Score >= a.Score ) as 'Rank'
FROM scores a
ORDER BY a.Score DESC


-- 262. 行程和用户
-- https://leetcode-cn.com/problems/trips-and-users/
SELECT
    request_at as 'Day', round(avg(Status!='completed'), 2) as 'Cancellation Rate'
FROM 
    trips t JOIN users u1 ON (t.client_id = u1.users_id AND u1.banned = 'No')
    JOIN users u2 ON (t.driver_id = u2.users_id AND u2.banned = 'No')
WHERE	
    request_at BETWEEN '2013-10-01' AND '2013-10-03'
GROUP BY 
    request_at


-- 1777. 每家商店的产品价格
-- https://leetcode-cn.com/problems/products-price-for-each-store/
select 
    product_id, 
    sum(case when store='store1' then price end) as store1, 
    sum(case when store='store2' then price end) as store2, 
    sum(case when store='store3' then price end) as store3
from products
group by product_id;


-- 181. 超过经理收入的员工
-- https://leetcode-cn.com/problems/employees-earning-more-than-their-managers/
SELECT a.Name AS Employee 
FROM   
    Employee  a, Employee b
WHERE
    a.ManagerId = b.Id AND a.Salary > b.Salary


-- 196. 删除重复的电子邮箱
-- https://leetcode-cn.com/problems/delete-duplicate-emails/
DELETE
    a
FROM 
    Person a, Person b
WHERE
    a.Email = b.Email AND a.Id > b.Id


-- 183. 从不订购的客户
-- https://leetcode-cn.com/problems/customers-who-never-order/
select customers.name as 'Customers'
from customers
where customers.id not in 
(
    select customerid from orders
)


-- 627. 变更性别
-- https://leetcode-cn.com/problems/swap-salary/
update salary set sex=IF(sex='f','m','f')


-- 626. 换座位https://leetcode-cn.com/problems/reformat-department-table/
-- https://leetcode-cn.com/problems/exchange-seats/
select 
    (
        case
            when mod(id, 2) != 0 and counts != id then id +1
            when mod(id, 2) != 0 and counts = id then id
            else id -1 
        end
    ) as id, student
from 
    seat, (select count(*) as counts from seat) as tmp
order by id ASC


-- 197. 上升的温度
-- https://leetcode-cn.com/problems/rising-temperature/
select
    a.id as "Id"
from 
    weather a join weather b 
    on datediff(a.recordDate, b.recordDate) = 1 and a.Temperature > b.Temperature


-- 601. 体育馆的人流量
-- https://leetcode-cn.com/problems/human-traffic-of-stadium/
select distinct t1.*
from stadium t1, stadium t2, stadium t3
where t1.people >= 100 and t2.people >= 100 and t3.people >= 100
and
(
	  (t1.id - t2.id = 1 and t1.id - t3.id = 2 and t2.id - t3.id =1)  -- t1, t2, t3
    or
    (t2.id - t1.id = 1 and t2.id - t3.id = 2 and t1.id - t3.id =1) -- t2, t1, t3
    or
    (t3.id - t2.id = 1 and t2.id - t1.id =1 and t3.id - t1.id = 2) -- t3, t2, t1
)
order by t1.id


-- 182. 查找重复的电子邮箱
-- https://leetcode-cn.com/problems/duplicate-emails/
select Email
from Person
group by Email
having count(Email) > 1


-- 1179. 重新格式化部门表
-- https://leetcode-cn.com/problems/reformat-department-table/
SELECT id, 
    SUM(CASE WHEN month='Jan' THEN revenue END) AS Jan_Revenue,
    SUM(CASE WHEN month='Feb' THEN revenue END) AS Feb_Revenue,
    SUM(CASE WHEN month='Mar' THEN revenue END) AS Mar_Revenue,
    SUM(CASE WHEN month='Apr' THEN revenue END) AS Apr_Revenue,
    SUM(CASE WHEN month='May' THEN revenue END) AS May_Revenue,
    SUM(CASE WHEN month='Jun' THEN revenue END) AS Jun_Revenue,
    SUM(CASE WHEN month='Jul' THEN revenue END) AS Jul_Revenue,
    SUM(CASE WHEN month='Aug' THEN revenue END) AS Aug_Revenue,
    SUM(CASE WHEN month='Sep' THEN revenue END) AS Sep_Revenue,
    SUM(CASE WHEN month='Oct' THEN revenue END) AS Oct_Revenue,
    SUM(CASE WHEN month='Nov' THEN revenue END) AS Nov_Revenue,
    SUM(CASE WHEN month='Dec' THEN revenue END) AS Dec_Revenue
FROM department
GROUP BY id
ORDER BY id;


-- 569. 员工薪水中位数
-- https://leetcode-cn.com/problems/median-employee-salary/
SELECT 
    Id, Company, Salary
FROM
    (SELECT 
        e.Id,
        e.Salary,
        e.Company,
        IF(@prev = e.Company, @Rank:=@Rank + 1, @Rank:=1) AS rank,
        @prev:=e.Company
    FROM
        Employee e, (SELECT @Rank:=0, @prev:=0) AS temp
    ORDER BY e.Company , e.Salary , e.Id) Ranking
        INNER JOIN
    (SELECT 
        COUNT(*) AS totalcount, Company AS name
    FROM
        Employee e2
    GROUP BY e2.Company) companycount ON companycount.name = Ranking.Company
WHERE
    Rank = FLOOR((totalcount + 1) / 2) OR Rank = FLOOR((totalcount + 2) / 2)
















































