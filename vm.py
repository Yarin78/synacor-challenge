from collections import defaultdict, deque
from queue import Queue, Empty
import concurrent.futures
import threading
import functools
import logging
import sys
import copy
import hashlib

logger = logging.getLogger(__name__)

# Behaviour when trying to read and there is nothing on the input
# Default is to pause the machine (and not update the IP)
CRASH_ON_EOF = False   # Raise exception if trying to read from input when there is no data
BLOCK_ON_EOF = False   # Block calling thread if trying to read from input when there is no data

SHOW_PROGRESS = False

OPCODE_HALT = 0
OPCODE_SET = 1
OPCODE_PUSH = 2
OPCODE_POP = 3
OPCODE_EQ = 4
OPCODE_GT = 5
OPCODE_JMP = 6
OPCODE_JT = 7
OPCODE_JF = 8
OPCODE_ADD = 9
OPCODE_MULT = 10
OPCODE_MOD = 11
OPCODE_AND = 12
OPCODE_OR = 13
OPCODE_NOT = 14
OPCODE_RMEM = 15
OPCODE_WMEM = 16
OPCODE_CALL = 17
OPCODE_RET = 18
OPCODE_OUT = 19
OPCODE_IN = 20
OPCODE_NOOP = 21

class Program(object):

    opcodes = {}
    ip_trace = deque()


    def __init__(self, prog_id=0):
        self.reset()
        self.prog_id = prog_id
        self._input = None
        self._output = None

    def reset(self):
        self.mem = [0] * 32768
        self.regs = [0] * 8
        self.stack = deque()

        self.ip = 0
        self.count = 0  # num instructions executed
        self.instr_count = [0]*len(self.mem)
        self.halted = False
        self.blocked_on_input = False
        self.last_in = 0
        self.last_out = 0

    def log_debug(self):
        logging.basicConfig(level=logging.DEBUG)

    def log_info(self):
        logging.basicConfig(level=logging.INFO)

    def log_warn(self):
        logging.basicConfig(level=logging.WARNING)

    def read(self, addr):
        assert addr >= 0 and addr < 32768
        data = self.mem[addr]
        assert data >= 0 and data < 65536
        logger.debug('Reading {} from [{}]'.format(data, addr))
        return data

    def read_number(self, num):
        assert num >= 0 and num < 32776
        return num if num < 32768 else self.regs[num - 32768]        

    def write(self, addr, data):
        assert addr >= 0 and addr < 32768
        logger.info('Writing {} to [{}]'.format(data, addr))
        self.mem[addr] = data

    def init_io(self, input=None, output=None):
        if input is None:
            self._input = StdinSource()
            print('Input from stdin')
        elif isinstance(input, list):
            self._input = Queue()
            for x in input:
                self._input.put(x)
        else:
            self._input = input

        if output is None:
            self._output = StdoutSink()
            print('Output to stdout')
        else:
            self._output = output

    def feed_input(self, v):
        assert isinstance(self._input, Queue)
        self._input.put(v)
        self.blocked_on_input = False

    def intercept(self):
        return False

    def load_data(self, addr, data):
        for ofs, v in enumerate(data):
            self.write(addr + ofs, v)

    def run(self, input=None, output=None, steps=0, breakpoints=None):
        self.init_io(input, output)
        return self._run(steps, breakpoints)

    def run_until_halted(self):
        try:
            while True:
                self.step()
        except MachineHaltedException:
            pass

    def start_async(self, start=0, daemon=False):
        global BLOCK_ON_EOF
        BLOCK_ON_EOF = True
        if self._input is None:
            self.init_io(Queue(), Queue())

        t = threading.Thread(target=self.run_until_halted, daemon=daemon, args=(start,))
        t.start()
        return t

    def _run(self, steps=0, breakpoints=None):
        if steps:
            while steps > 0 and self.step():
                steps -= 1
        elif breakpoints:
            while self.ip not in breakpoints:
                self.step()
        else:
            while self.step():
                pass

        if isinstance(self._output, ReturnSink):
            return self._output.values

    def run_until_next_io(self, input=None, output=None, feed_input=None):
        if self.count == 0:
            self.init_io(input if input else Queue(), output if output else Queue())
        if feed_input:
            for x in feed_input:
                self.feed_input(x)
        while not self.halted and not self.blocked_on_input and self._output.empty():
            self.step()
        if self.halted:
            print('HALT')
        if self.halted or self.blocked_on_input:
            return None
        return self._output.get()

    def read_token(self):
        '''Reads a token from the programs output'''
        s = ''
        while not self.halted and not self.blocked_on_input:
            while not self._output.empty():
                c = self._output.get()
                if c in [10, 32]:
                    if s:
                        return s
                else:
                    if c < 32 or c > 127:
                        assert not s
                        # non-ASCII values are returned as-is
                        return c
                    s += chr(c)
            self.step()
        return None

    def read_line(self):
        s = ''
        while (not self.halted and not self.blocked_on_input) or not self._output.empty():
            while not self._output.empty():
                c = self._output.get()
                if c == 10:
                    return s
                if c < 32 or c > 127:
                    assert not s
                    return "<%d>" % c
                s += chr(c)
            self.step()
        return None

    def write_line(self, line):
        for c in line:
            self._input.put(ord(c))
        self._input.put(10)

    def step(self):
        assert self.input and self.output
        if self.halted:
            raise MachineHaltedException()
        if not self.intercept():
            self.ip_trace.append(self.ip)
            if len(self.ip_trace) > 100:
                self.ip_trace.popleft()

            opcode = self.read(self.ip)            
            
            if opcode not in self.opcodes:
                raise UnknownOpcodeException('unknown opcode %d at addr %d' % (opcode, self.ip))
            (instr, mnemonic, length) = self.opcodes[opcode]
            params = [self.read(self.ip + ofs) for ofs in range(1, length)]                
            
            #if logger.isEnabledFor(logging.INFO):
            #    (mnemonic, code) = self.decode(self.ip)
            #    logger.info('%2d %5d: Executing %s' % (self.prog_id, self.ip, mnemonic))
            self.count += 1
            if SHOW_PROGRESS and self.count % 10000 == 0:
                sys.stderr.write('.')
                sys.stderr.flush()
            self.instr_count[self.ip] += 1
            default_new_ip = self.ip + length
            new_ip = instr(self, *params)
            self.ip = default_new_ip if new_ip is None else new_ip  # Must distinguish 0 and None
        return not self.halted

    def show(self, addr, end_addr):
        #try:
            while addr < end_addr:
                opcode = self.mem[addr]                
                opcode_addr = addr
                if opcode in self.opcodes:
                    (_, mnemonic, length) = self.opcodes[opcode]
                    params = [self.read(addr + ofs) for ofs in range(1, length)]                
                    code = mnemonic
                    code += '  ' + ','.join(['0x%04x' % p for p in params])
                    addr += length
                else:
                    code = 'DB 0x%04x' % self.mem[addr]
                    addr += 1

                line = '%04x  %-30s' % (opcode_addr, code)
                if self.instr_count[opcode_addr]:
                    line += '[%6d]' % (self.instr_count[opcode_addr])
                print(line)
        #except:
        #    pass

    def hotspots(self):
        # Might want to use show instead
        for i in range(len(self.instr_count)):
            if self.instr_count[i] > 0:
                print('%5d %15d' % (i, self.instr_count[i]))

    def get_state(self):
        words = [*self.mem, *self.regs, self.ip, *self.stack]
        state = bytearray()
        for word in words:
            state.append(word % 256)
            state.append(word // 256)
        return state
    
    def set_state(self, bytes):        
        words = [bytes[i] + bytes[i+1] * 256 for i in range(0, len(bytes), 2)]
        self.mem = words[0:32768]
        self.regs = words[32768:32768+8]
        self.ip = words[32768+8]
        self.stack = deque(words[32768+9:])

    def save_state(self):
        state = self.get_state()
        filename = f"states/{hashlib.md5(state).hexdigest()}.bin"
        with open(filename, "wb") as f:
            f.write(state)
        print(f"Saving state to {filename}")

    def load_state(self, filename):
        with open(filename, "rb") as f:
            state = f.read()
        self.set_state(state)

    def input(self):
        if BLOCK_ON_EOF:
            try:
                self.last_in = self._input.get()
                self.blocked_on_input = False
                return self.last_in
            except Empty:
                # If we have multiple queues as input, this can happen because
                # we have no good way of blocking.
                self.blocked_on_input = True
                return None
        elif CRASH_ON_EOF:
            self.last_in = self._input.get_nowait()
            self.blocked_on_input = False
            return self.last_in
        else:
            try:
                if self._input.is_empty():
                    self.save_state()
                self.last_in = self._input.get_nowait()
                #logger.info('%2d        Read %d' % (self.prog_id, self.last_in))
                self.blocked_on_input = False
                return self.last_in
            except Empty:
                #logger.info('%2d        Blocked' % (self.prog_id))
                self.blocked_on_input = True
                return None

    def output(self, value):
        self.last_out = value
        self._output.put(value)

    # If an opcode returns a non-value, it's the value of the new IP
    # Otherwise the length of the opcode is added to the IP

    def opcode_halt(self):
        self.halted = True
        return self.ip

    def opcode_set(self, a, b):        
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = self.read_number(b)        

    def opcode_push(self, a):
        self.stack.append(self.read_number(a))

    def opcode_pop(self, a):
        assert len(self.stack) > 0
        assert a >= 0x8000 and a < 0x8008
        v = self.stack.pop()        
        self.regs[a-0x8000] = v

    def opcode_eq(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = 1 if self.read_number(b) == self.read_number(c) else 0

    def opcode_gt(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = 1 if self.read_number(b) > self.read_number(c) else 0

    def opcode_jmp(self, a):
        return self.read_number(a)

    def opcode_jt(self, a, b):
        if self.read_number(a) != 0:
            return self.read_number(b)

    def opcode_jf(self, a, b):
        if self.read_number(a) == 0:
            return self.read_number(b)

    def opcode_add(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = (self.read_number(b) + self.read_number(c)) % 32768

    def opcode_mult(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = (self.read_number(b) * self.read_number(c)) % 32768

    def opcode_mod(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = (self.read_number(b) % self.read_number(c)) % 32768

    def opcode_and(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = (self.read_number(b) & self.read_number(c)) % 32768

    def opcode_or(self, a, b, c):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = (self.read_number(b) | self.read_number(c)) % 32768

    def opcode_not(self, a, b):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = 32767 - self.read_number(b)

    def opcode_rmem(self, a, b):
        assert a >= 0x8000 and a < 0x8008
        self.regs[a-0x8000] = self.read(self.read_number(b))

    def opcode_wmem(self, a, b):
        self.write(self.read_number(a), self.read_number(b))

    def opcode_call(self, a):
        self.stack.append(self.ip + 2)
        return self.read_number(a)

    def opcode_ret(self):
        if len(self.stack) == 0:
            self.halted = True
            return self.ip
        return self.stack.pop()

    def opcode_out(self, a):
        self.output(self.read_number(a))

    def opcode_in(self, a):
        assert a >= 0x8000 and a < 0x8008
        value = self.input()
        if value is None:
            return self.ip
        self.regs[a-0x8000] = value

    def opcode_noop(self):
        pass

    def opcode_debug(self, a):
        print('DEBUG', a)

    # (function, mnemonic, length)
    opcodes = {
        OPCODE_HALT: (opcode_halt, 'HALT', 1),
        OPCODE_SET: (opcode_set, 'SET', 3),
        OPCODE_PUSH: (opcode_push, 'PUSH', 2),
        OPCODE_POP: (opcode_pop, 'POP', 2),
        OPCODE_EQ: (opcode_eq, 'EQ', 4),
        OPCODE_GT: (opcode_gt, 'GT', 4),
        OPCODE_JMP: (opcode_jmp, 'JMP', 2),
        OPCODE_JT: (opcode_jt, 'JT', 3),
        OPCODE_JF: (opcode_jf, 'JF', 3),
        OPCODE_ADD: (opcode_add, 'ADD', 4),
        OPCODE_MULT: (opcode_mult, 'MULT', 4),
        OPCODE_MOD: (opcode_mod, 'MOD', 4),
        OPCODE_AND: (opcode_and, 'AND', 4),
        OPCODE_OR: (opcode_or, 'OR', 4),
        OPCODE_NOT: (opcode_not, 'NOT', 3),
        OPCODE_RMEM: (opcode_rmem, 'RMEM', 3),
        OPCODE_WMEM: (opcode_wmem, 'WMEM', 3),
        OPCODE_CALL: (opcode_call, 'CALL', 2),
        OPCODE_RET: (opcode_ret, 'RET', 1),
        OPCODE_OUT: (opcode_out, 'OUT', 2),
        OPCODE_IN: (opcode_in, 'IN', 2),
        OPCODE_NOOP: (opcode_noop, 'NOOP', 1),
    }


def wire_up_serial(programs, input, output):
    '''Connects multiple programs with each other in a sequence.'''
    pipes = [Queue() for _ in range(len(programs) - 1)]
    for i in range(len(programs)):
        programs[i].init_io(pipes[i-1] if i > 0 else input, pipes[i] if i < len(programs) - 1 else output)


def parallel_executor(programs):
    '''Executes one instruction at a time across all programs in round robin fashion,
    until they're all halted. Assumes the IO has already been setup.
    Returns the an array, one element per input program. If the output is a ReturnSink
    for a program, the corresponding element will contain that list, otherwise None.
    '''
    global BLOCK_ON_EOF
    BLOCK_ON_EOF = False

    while True:
        all_halted = True
        all_blocked = True
        for prog in programs:
            if not prog.halted:
                prog.step()
                if not prog.blocked_on_input:
                    all_blocked = False
                all_halted = False
        if all_halted:
            break
        if all_blocked:
            raise MachineBlockedException()

    result = []
    for prog in programs:
        if isinstance(prog._output, ReturnSink):
            result.append(prog._output.values)
        else:
            result.append(None)
    return result

def threaded_executor(programs):
    '''Executes all programs in separate threads until they're all halted.
    Returns the an array, one element per input program. If the output is a ReturnSink
    for a program, the corresponding element will contain that list, otherwise None.'''
    global BLOCK_ON_EOF
    BLOCK_ON_EOF = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(programs)) as executor:
        result = executor.map(lambda p: p._run(), programs)
    return list(result)


class MachineHaltedException(Exception):
    '''Thrown when trying to execute code while the machine is halted.'''
    pass

class MachineBlockedException(Exception):
    '''Thrown when trying to execute code and the machine is blocked on input.'''
    pass

class ProgramOutOfBoundsException(Exception):
    '''Tried to execute an instruction outside of the programs memory.'''
    pass

class UnknownOpcodeException(Exception):
    '''Tried to execute an unknown opcode.'''
    pass

class ReturnSink(object):
    def __init__(self):
        self.values = []

    def put(self, x):
        self.values.append(x)

class StdoutSink(object):
    def put(self, x):
        if x >= 32 and x < 127:
            sys.stdout.write(chr(x))
        elif x == 10:
            sys.stdout.write('\n')
        else:
            sys.stdout.write(str(x) + '\n')

class StdinSource(object):
    def __init__(self):
        self._queue = Queue()

    def is_empty(self):
        return self._queue.empty()

    def get(self, nowait=False):
        if self._queue.empty():
            s = input()
            if s == '':
                if nowait:
                    raise Empty()
                else:
                    raise MachineBlockedException()
            for c in s:
                self._queue.put(ord(c))
            self._queue.put(10)
        return self._queue.get()

    def get_nowait(self):
        return self.get(True)

class JoinedSource(object):
    '''Gets input from multiple sources.'''
    def __init__(self, queues):
        self.queues = queues

    def get(self):
        # No good way of doing a blocking get here
        return self.get_nowait()

    def get_nowait(self):
        # Need multiprocessing queues here
        for q in self.queues:
            try:
                return q.get_nowait()
            except Empty:
                pass
        raise Empty()

class DuplicateSink(object):
    '''Sends the same output to multiple output sources.'''
    def __init__(self, queues):
        self.queues = queues

    def put(self, x):
        for q in self.queues:
            q.put(x)

class BaseInput(object):
    def get(self):
        raise Exception()  # Should be overridden

    def get_nowait(self):
        return self.get()


class PythonProgram:
    pass

if __name__ == "__main__":
    SHOW_PROGRESS = False
    with open(sys.argv[1], "r") as f:
        code = f.readline()
    prog = Program(code)
    prog.run(input=StdinSource(), output=StdoutSink())
