open("/Users/alice/Desktop/stage_2026/python/spectrogrammes/originaux/R_260730_Letter_1_p.avi");
makeRectangle(622, 516, 11, 180);
print("A");

run("Duplicate...", "title=copy duplicate");

selectWindow("copy");
print("AA")
run("Reslice [/]...", "output=1.000 start=Left rotate");
print("B");
run("AVI... ", "compression=None frame=60 save=/Users/alice/Desktop/stage_2026/python/spectrogrammes/test3.avi");
print("D")
