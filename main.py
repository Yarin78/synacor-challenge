from vm import *


p = Program()

p.init_io()

with open("challenge.bin", "rb") as f:
    data = f.read()

data = [data[i*2] + data[i*2+1] * 256 for i in range(len(data) // 2)]
p.load_data(0, data)

# Init code: pqrqDkzUdcBq
# Start code: PYFwxelibmjc
# Self test code: ZjBewmzZXgGc
# Use tablet code: skUlxNCWefPr
# Code walking in maze GvybMzUhoFux
# Use teleporter after coins: VkAYDPTOSmJt
# Use teleporter after patching registers: WvdDEWYFDxSg
# Code in mirror: vpAMxUXbxMYV  -> VYMxdXUxMAqv

# In maze at ladder with lantern and tablet
#p.load_state("states/ce9e914ea2447d33975ab7634bf06b76.bin")

# In 2392 with lantern, tablet and can
#p.load_state("states/730dae3e34a894893257f25ad5ee095a.bin")

# In 2447, ruins
#p.load_state("states/2966542b0cc7511848e3b93e9aa1a119.bin")

# In Synacor Headquarters
#p.load_state("states/43a812ceb89049f19d2195ac75740461.bin")
#p.load_state("states/f1c5c62d050f705b7985d2774b8a84bf.bin")

#p.load_state("states/65ae8ef111ffa67df920a3a3dfbf0bb8.bin")

# Patching ackermann function check
# p.regs[7] = 25734
# p.mem[0x156b] = 1
# p.mem[0x156b+1] = 0x8000
# p.mem[0x156b+2] = 6
# p.mem[0x1571] = 21
# p.mem[0x1572] = 21
#print(p.regs)

# Challenge done
# p.load_state("states/a93c66df47fe4a19d5ba7fa5e3066083.bin")
# Code in mirror: vpAMxUXbxMYV


#p.log_info()

# 6027-6067
# p.show(0x156b, 0x156b+20)
# p.run(breakpoints=list(range(6027, 6068)))
# v = list(p.ip_trace)
# for i in range(-20, 0):
#     p.show(p.ip_trace[i], p.ip_trace[i]+1)
# #p.hotspots()

p.run()
