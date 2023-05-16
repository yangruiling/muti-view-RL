from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.raw_utils import encode_multi_smiles_list
from utils.trans_dict import get_vac_size
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


text_coder_config = {
    "embed_dim": 32,
    "hid_dim": 128,
    "out_dim": 64,
    "num_layer": 2,
    "max_len": 198,
}


class SeqEncoder(nn.Module):
    def __init__(self, max_len, config):
        super(SeqEncoder, self).__init__()

        self.emb_dim = config["embed_dim"]
        self.hid_dim = config["hid_dim"]
        self.out_dim = config["out_dim"]
        self.num_layer = config["num_layer"]
        self.max_len = max_len

        # here `146` is the length of tran_dict _a2i
        self.emb = torch.nn.Embedding(250, self.emb_dim) # （词典维数，嵌入维数）
        # self.emb = torch.nn.Embedding(get_vac_size() + 10, self.emb_dim)

        self.lin1 = nn.Linear(self.emb_dim, self.hid_dim)

        # remind: the num_layer will increase the risk of optimizer becoming NAN
        self.rnn = nn.GRU(
            input_size=self.hid_dim,
            num_layers=self.num_layer,
            hidden_size=self.hid_dim,
            bidirectional=False,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hid_dim * self.max_len, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hid_dim, self.out_dim),
        )

    def forward(self, x, length):
        batch_size = x.size(0)
        embed_x = self.lin1(self.emb(x))
        # print("embed size {}".format(embed_x.shape))

        cur_lens = length.data
        for i, len in enumerate(cur_lens) :
            if len > self.max_len:
                cur_lens[i] = torch.tensor(self.max_len)
                
        cur_lens = cur_lens.to("cpu")
        # 通常pading的位置向量都是0，我们需要使用pack_padded_sequence()把数据压紧，即去掉pading的部分，减少冗余。
        # 然后再输入网络中，如lstm等。
        # 通过网络后的结果也是压紧的，需要通过pad_packed_sequence()还原。
        packed_embed_x = pack_padded_sequence(
            embed_x, # 特征向量
            batch_first=True, # the input is expected in B x T x * format
            lengths=cur_lens, # 每个Tensor的实际长度
            #   lengths=length,
            enforce_sorted=False,
        )

        packed_embed_x, _ = self.rnn(packed_embed_x)

        x = pad_packed_sequence(
            packed_embed_x, batch_first=True, total_length=self.max_len
        )[0]

        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x
