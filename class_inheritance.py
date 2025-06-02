class Student:
    def __init__(self, id, name, age, gender, department):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.department = department

    def show(self):
        print("====== 학생 정보 ======")
        print(f"학번 : {self.id}")
        print(f"이름 : {self.name}")
        print(f"나이 : {self.age}")
        print(f"성별 : {self.gender}")
        print(f"학과 : {self.department}")

    def add_age(self, offset):
        self.age += offset

student1 = Student("200000", "홍길동", 20, "남성", "컴퓨터공학과")
student2 = Student("232100", "조정빈", 25, "남성", "소프트웨어학과")
student3 = Student("200532", "표창원", 60, "남성", "교통에너지융합학과")

student1.show()
student2.show()
student3.show()

class Jungbin(Student):
    def __init__(self, id, name, age, gender, department, subject):
        super().__init__(id, name, age, gender, department)
        self.subject = subject

    def show_jungbin(self):
        print("----- jungbin information -----")
        print(f"jungbin's name : {self.name}")
        print(f"jungbin's age : {self.age}")
        print(f"jungbin's subject : {self.subject}")

jojungbin = Jungbin("232100", "조정빈", 25, "남성", "소프트웨어학과", "소프트웨어")
jojungbin.show_jungbin()
