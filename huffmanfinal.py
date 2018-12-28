"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    >>> d = make_freq_dict(bytes([65, 66, 67, 66, 67]))
    >>> d == {65:1, 66:2, 67:2}
    True
    >>> d = make_freq_dict(bytes([]))
    >>> d == {}
    True
    """
    i = 0
    dic = {}
    while i < len(text):
        if not text[i] in dic:
            dic[text[i]] = 1
        else:
            dic[text[i]] += 1
        i += 1
    return dic


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq = {2:6, 3:4, 4:1}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(3), HuffmanNode(4)), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(None,\
HuffmanNode(4), HuffmanNode(3)), HuffmanNode(2))
    >>> t == result1 or t == result2
    True
    >>> freq = {}
    >>> t = huffman_tree(freq)
    >>> result = HuffmanNode(None, None, None)
    >>> t == result
    True
    """

    if not isinstance(freq_dict, dict):
        return None

    if len(freq_dict) == 1:
        symbol = list(freq_dict)
        tree = HuffmanNode(None, HuffmanNode(symbol[0]), None)
        return tree

    holder = []
    min_list = []
    check = 0
    huff = HuffmanNode(None, None, None)
    for (key, value) in freq_dict.items():
        holder.append((value, key))
    holder = sorted(holder)
    while len(holder) != 0:
        if len(holder) > 1:
            m_left = holder.pop(0)
            h_right = holder.pop(0)
            min_list.append(HuffmanNode \
                                (None, HuffmanNode(m_left[1]), HuffmanNode(h_right[1])))
        elif len(holder) == 1:
            h_right = holder.pop(0)
            min_list.append(HuffmanNode((h_right[1])))

    if len(min_list) == 1:
        return min_list[0]
    else:
        while check < len(min_list):
            if check == 0:
                huff = build_tree(min_list[check], min_list[check + 1])
                check += 1
            else:
                huff = build_tree(huff, min_list[check])
            check += 1
    return huff


def build_tree(tree1, tree2, root=None):
    """
    :param list:
    :return:
    """
    return HuffmanNode(root, tree1, tree2)


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3),\
HuffmanNode(2)), HuffmanNode(None, HuffmanNode(4), HuffmanNode(5)))
    >>> d = get_codes(tree)
    >>> d == {3: "00", 2:"01", 4:"10", 5:"11"}
    True
    """
    if not isinstance(tree, HuffmanNode):
        return None

    x = node(tree)
    final = {}
    for item in x:
        final[item] = item_code(tree, item)

    return final


def item_code(tree, item):  # get code helper
    '''
    returns a string representing the code of the symbol in the tree
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param symbol object
    @rtype: str

    '''
    x = item
    code = ''
    if not tree:
        return None
    if tree.symbol == x:
        return code
    else:
        try:
            code = code + '0' + item_code(tree.left, x)
            return code
        except TypeError:  # cant convert to nonetype object implicitly
            pass
        try:
            code = code + '1' + item_code(tree.right, x)
            return code
        except TypeError:  # cant convert to nonetype object implicitly
            pass


def node(tree):  # get code helper
    '''
    returns a list that contains all the symbols within the tree
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: list(object)

    '''

    if not tree:
        return []
    if tree.is_leaf():
        return [tree.symbol]
    else:
        return node(tree.left) + node(tree.right)


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> tree = HuffmanNode(None, left, None)
    >>> number_nodes(tree)
    >>> tree.number
    1

    """

    if not isinstance(tree, HuffmanNode):
        return None
    i = 0
    check = number_node_help(tree, i)
    if check == 1:
        tree.number = 0
        return None
    elif check == 2:
        if isinstance(tree.left, HuffmanNode):
            l = tree.left
            l.number = 0
            tree.left = l
            tree.number = 1
            return None
        else:
            r = tree.right
            r.number = 0
            tree.right = r
            tree.number = 1
            return None
    else:
        if tree.right.right or tree.right.left:
            z = tree.right
            tree.right = HuffmanNode(00, None, None)
            assign(tree, check - 1)
            display(tree)
            x = tree.number
            z.number = x - 1
            tree.right = z
        else:
            assign(tree, check, )
            display(tree)
            tree.number -= 1
            if tree.right and tree.right.number:
                x = tree.right
                x.number = None
                tree.right = x


def assign(tree, total):
    """ Assigns the left node.
       @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
       @param int
       @rtype: NoneType
       """
    if not tree:
        return None
    if tree.symbol is None:
        tree.number = total
    assign(tree.left, total - 2)


def display(tree):  # assigns the right
    '''
    fills in all the right nodes in the sub tree
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: None
    '''

    if not tree:
        return None

    display(tree.right)
    display(tree.left)

    if tree.symbol is None and tree.number and not tree.right.number:
        z = tree.right
        x = tree.number - 1
        z.number = x
        tree.right = z


def number_node_help(tree, i):
    '''
    Calculates the sum of all the internal nodes
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param int:
    @rtype: int
    '''
    x = i
    if not tree:
        return None
    try:
        x += number_node_help(tree.left, i)

    except TypeError:  # int + nonetype
        pass
    try:
        x += number_node_help(tree.right, i)
    except TypeError:  # int + nonetype
        pass
    if tree.symbol is None:
        x += 1
        return x


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    >>> freq = {3:2, 2:7, 9:1}
    >>> tree = huffman_tree(freq)
    >>> avg_length(tree, freq)
    1.3
    """

    if not (isinstance(tree, HuffmanNode) or isinstance(freq_dict, dict)):
        raise TypeError

    codes = get_codes(tree)
    total = 0

    for item in codes:
        x = len(codes[item])
        y = freq_dict[item]
        total += x * y
    total1 = 0
    for item in freq_dict:
        total1 += freq_dict[item]

    final = 0
    final = total / total1

    return final


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text1 = "jump of a building"
    >>> d = make_freq_dict(text1)
    >>> tree = huffman_tree(d)
    >>> codes = get_codes(tree)
    >>> result = generate_compressed(text1, codes)
    >>> [byte_to_bits(byte) for byte in result]
    ['00001110', '00011010', '11001100', '00011110', '00000011', \
'00000011', '00110001', '00000010', '01100100', '00010000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> d = {65: "1111", 66: "101", 68: "100",\
     69: "1110", 70: "1101", 200: "1100", 90: "0"}
    >>> text = bytes([68,68,66,69,70,90,200,66,90,90,90,90,90,90,90,90,90,65])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10010010', '11110110', '10110010', '10000000', '00111100']
    """

    final = []
    txt = ''
    i = 0
    for item in text:
        i += 1
        txt += codes[item]
        if len(txt) == 8:
            final.append(bits_to_byte(txt))
            txt = ""
        if len(txt) > 8:
            while len(txt) > 8:
                final.append(bits_to_byte(txt[:8]))
                txt = txt[8:]

        if i == len(text) and len(txt) < 8:
            final.append(bits_to_byte(txt[:8]))

    return bytes(final)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
HuffmanNode(2)), HuffmanNode(None, HuffmanNode(5), HuffmanNode(4)))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 0, 5, 0, 4, 1, 0, 1, 1]
    """

    # conditions:
    # internal nodes take upto four bytes ####
    # 1: if left subtree is a leaf then 0 | else 1
    # 2: if left subtree is a leaf then symbol| else node-number
    # 3: if right subtree is a leaf then 0 | else 1
    # 4: if right subtree is a leaf then symbol | else node-number

    if not isinstance(tree, HuffmanNode):
        return None

    text = ''
    code = byte_code(tree, text)
    byte_repr = bytes([])
    sp = code.split()
    codes = []
    items = sp[:]
    i = 0
    while i < len(sp):
        items.remove(sp[i])
        codes.append(sp[i])
        i += 2
    i = 0
    while i < len(codes):
        byte_repr += bytes([int(codes[i])])
        byte_repr += bytes([int(items[i])])
        i += 1
    return byte_repr


def byte_code(tree, code):
    """
    returns the byte code in string format
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param empty string
    @rtype: string

    """
    x = code
    code = ''
    if not tree:
        return None
    if tree.left:
        code += byte_code(tree.left, x)

    if tree.right:
        code += byte_code(tree.right, x)

    if isinstance(tree.left, HuffmanNode):
        if tree.left.is_leaf():
            code += '0'
            code += (' ' + str(tree.left.symbol) + ' ')
        else:
            code += '1'
            code += (' ' + str(tree.left.number) + ' ')

    if isinstance(tree.right, HuffmanNode):
        if tree.right.is_leaf():
            code += '0'
            code += ' ' + str(tree.right.symbol) + ' '
        else:
            code += '1'
            code += (' ' + str(tree.right.number) + ' ')
    return code


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    >>> lst = [ReadNode(0,5,0,7)]
    >>> generate_tree_general(lst, 0)
    HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None))
    """
    # basically making the tree using the concept in tree_to_bytes

    # conditions:
    # internal nodes take upto four bytes ####
    # 1: if left subtree is a leaf then 0 | else 1
    # 2: if left subtree is a leaf then symbol| else node-number
    # 3: if right subtree is a leaf then 0 | else 1
    # 4: if right subtree is a leaf then symbol | else node-number

    # the numbe of the node corresponds to its place in the list

    '''
    l_type: 0/1 (if the corresponding HuffmanNode's left is a leaf)
    l_data: a symbol or the node number of a HuffmanNode's left
    r_type: 0/1 (if the corresponding HuffmanNode's right is a leaf)
    r_data: a symbol or the node number of a HuffmanNode's righ
    '''

    if not (isinstance(root_index, int) or (isinstance(node_lst, list))):
        return None
    if len(node_lst) == 1:
        singletree = make_node(node_lst[0])
        singletree.number = 0
        return singletree

    internal_nodes = []
    i = 0
    while i < len(node_lst):
        x = make_node(node_lst[i])
        x.number = i
        internal_nodes.append(x)
        i += 1

    root = internal_nodes[root_index]
    internal_nodes[root_index] = 'Nothing'

    i = 0
    while i < len(internal_nodes):
        for x in range(len(internal_nodes)):
            if x != root_index:
                tree = combine(root, x, internal_nodes)
                if tree:
                    root = tree
                    i += 1

    return root


def combine(tree, nodes_num, node_lst):
    """ To find nod e_num in tree
        @param A huffman tree
        @param int
        @rtype: list
        """
    if not tree:
        return None

    try:
        if tree.left.number == nodes_num:
            tree.left = node_lst[nodes_num]
            return tree
    except AttributeError:
        pass
    try:
        if tree.right.number == nodes_num:
            tree.right = node_lst[nodes_num]
            return tree
    except AttributeError:
        pass
    combine(tree.left, nodes_num, node_lst)
    combine(tree.right, nodes_num, node_lst)


def make_node(read_node):
    '''
    Returns a huffman node based on the properties of the readNode object
    @param Read_Node
    @rtype HuffmanNode
    '''

    x = HuffmanNode(None)
    if read_node.l_type == 0:
        y = HuffmanNode(read_node.l_data)
        x.left = y
    else:
        y = HuffmanNode(None)
        y.number = read_node.l_data
        x.left = y
    if read_node.r_type == 0:
        y = HuffmanNode(read_node.r_data)
        x.right = y
    else:
        y = HuffmanNode(None)
        y.number = read_node.r_data
        x.right = y

    return x


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), HuffmanNode(None, \
HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), \
ReadNode(1, 0, 1, 0), ReadNode(0,5,0,6), ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 4)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, HuffmanNode\
(1, None, None), HuffmanNode(2, None, None)), HuffmanNode(None,\
 HuffmanNode(3, None, None), HuffmanNode(4, None, None))),\
 HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode\
(6, None, None)))
    >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(0, 2, 0, 30), \
ReadNode(1, 0, 1, 0), ReadNode(1,0,0,0) ,ReadNode(0,3,0,4), \
ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(1, None, None), \
HuffmanNode(2, None, None)), HuffmanNode(None, HuffmanNode(2, None, None), \
HuffmanNode(30, None, None)))
    """

    if not (isinstance(root_index, int) or isinstance(node_lst, list)):
        return None

    huffmanlist = []
    using_nodes = node_lst[:root_index + 1]
    while len(using_nodes) > 0:
        left_data = using_nodes[0].l_data
        right_data = using_nodes[0].r_data
        leftside = HuffmanNode(left_data)
        rightside = HuffmanNode(right_data)
        both = HuffmanNode(None, leftside, rightside)
        huffmanlist.append(both)
        using_nodes.pop(0)
    while len(huffmanlist) > 0:
        if len(huffmanlist) > 3:
            store1 = huffmanlist[0]
            store2 = huffmanlist[1]
            huffmanlist.pop(0)
            huffmanlist.pop(0)
            huffmanlist.pop(0)
            huffmanlist.insert(0, (HuffmanNode(None, store1, store2)))
        elif len(huffmanlist) == 3:
            store1 = huffmanlist[0]
            store2 = huffmanlist[1]
            return HuffmanNode(None, store1, store2)
        elif len(huffmanlist) == 2:
            return HuffmanNode(None, huffmanlist[0], huffmanlist[1])
        else:
            return huffmanlist[0]


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text1 = bytes([1, 2, 1, 0])
    >>> result1 = generate_compressed(text1, d)
    >>> tree1 = HuffmanNode(None, HuffmanNode(0),\
     HuffmanNode(None, HuffmanNode(1),  HuffmanNode(2)))
    >>> x = generate_uncompressed(tree1, result1, 4)
    >>> x == text1
    True
    >>> text2 = bytes([1, 2, 1, 0, 2])
    >>> result2 = generate_compressed(text2, d)
    >>> x = generate_uncompressed(tree1, result2, 5)
    >>> x == text2
    True

    """

    x = list([byte_to_bits(byte) for byte in text])
    codes = get_codes(tree)

    bit_code = ''
    for item in x:
        bit_code += item

    new_text = []
    c = ''
    i = 0
    symbols = list(codes.keys())
    co = list(codes.values())
    replace = bit_code[:]

    while i < len(bit_code):
        c += replace[i]
        if c in co and len(new_text) < size:
            y = co.index(c)
            new_text.append(symbols[y])

            c = ''

        i += 1
    return bytes(new_text)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    x = get_codes(tree)
    o_holder = []
    t_holder = []
    final = []
    i = 0

    for (key, value) in freq_dict.items():
        t_holder.append((value, key))
    for keys, values in x.items():
        o_holder.append((len(values), keys))

    t_holder = sorted(t_holder, reverse=True)
    o_holder = sorted(o_holder)

    while i < len(t_holder):
        final.append(((o_holder[i][1]), (t_holder[i][1])))
        i += 1

    for ch in final:
        if ch[0] != ch[1]:
            helper_recursion(tree, ch[0], ch[1])

    re_helper_recursion(tree)


def helper_recursion(tree, find, change):
    """
    The function goes through the nodes of the r\
    tree and finds the variable find and replaces with change.

    :param tree:
    :param change:
    :return:
    """
    if not tree:
        pass
    if tree.symbol == find:
        tree.symbol = [change]
    if tree.left:
        helper_recursion(tree.left, find, change)
    if tree.right:
        helper_recursion(tree.right, find, change)


def re_helper_recursion(tree):
    """
    checks if any symbol in the tree is a r\
    list and if it is takes the type list out
    :param tree:
    :return:
    """
    if not tree:
        pass
    if type(tree.symbol) == list:
        tree.symbol = tree.symbol[0]
    if tree.left:
        re_helper_recursion(tree.left)
    if tree.right:
        re_helper_recursion(tree.right)


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    import doctest

    doctest.testmod()
    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))