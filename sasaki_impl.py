import torch
import torch.nn as nn
import torch.nn.functional as F


class SasakiModel(nn.Module):
    def __init__(self, n_vocab_subword, embedding_dim):
        super(SasakiModel, self).__init__()
        self.n_vocab_subword = n_vocab_subword
        self.embedding_dim = embedding_dim
        self.q_embeddings = nn.Embedding(n_vocab_subword, embedding_dim)
        self.k_embeddings = nn.Embedding(n_vocab_subword, embedding_dim)
        self.v_embeddings = nn.Embedding(n_vocab_subword, embedding_dim)

    def get_word_vec_attn(self, k_idx: torch.Tensor, v_idx, q_idx):
        """
        :param k_idx: (batch_size, seq_len, emb_dim)
        :param v_idx:
        :param q_idx:
        :return:
        """
        batchsize, seq_len = k_idx.shape

        # not exactly sure what mask is doing here...
        mask = -F.relu(-k_idx) * 10000.0
        mask = mask.reshape((batchsize, seq_len)).repeat(1, 1, self.embedding_dim)

        k = self.k_embeddings(k_idx)
        q = self.q_embeddings(v_idx)
        v = self.v_embeddings(q_idx)

        q = torch.sum(q, dim=1, keepdim=True).repeat(1, seq_len, 1)
        a = F.softmax((q * k) * (self.embedding_dim ** 0.5) + mask, dim=1)

        return torch.sum(a * v)

    def forward(self, k_idx, v_idx, q_idx, ref_vector, freq):
        sub_vector = self.get_word_vec_attn(k_idx, v_idx, q_idx)

        sub_vector = F.normalize(sub_vector, p=2, dim=1)
        ref_vector = F.normalize(ref_vector, p=2, dim=1)

        sq_loss = torch.sum((sub_vector - ref_vector) ** 2 / self.embedding_dim, dim=1, keepdim=True)

        sq_loss = sq_loss * torch.log(freq)
        return 1 - sq_loss


model = SasakiModel(100, 100)
model.train()
for p in model.parameters():
    if p.dim() > 1:
        nn.init.normal_(p, mean=0, std=0.01)
        # nn.init.xavier_uniform_(p)

total_loss = 0
epochs = 300
print_every = 10

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
for epoch in range(epochs):
    for i, batch in enumerate(train_iter):
        src = batch.English.transpose(0,1)
        trg = batch.French.transpose(0,1)



        optim.zero_grad()

        loss = model(src, )

        loss.backward()
        optim.step()

        total_loss += loss.data[0]
        if (i + 1) % print_every == 0:
            loss_avg = total_loss / print_every
            print(loss_avg)