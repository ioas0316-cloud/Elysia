from Core.L1_Foundation.M1_Keystone.organism import cell

@cell(\"TestHeart\", sensitivity=0.9)
class TestHeart:
    def beat(self):
        print(\"Heart is beating from the Void.\")
