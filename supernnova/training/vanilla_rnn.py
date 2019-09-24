import torch
import torch.nn.functional as F


class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size, settings):
        super(VanillaRNN, self).__init__()

        # Params
        self.layer_type = settings.layer_type
        self.output_size = settings.nb_classes
        self.hidden_size = settings.hidden_dim
        self.num_layers = settings.num_layers
        self.dropout = settings.dropout
        self.bidirectional = settings.bidirectional
        self.use_cuda = settings.use_cuda
        self.rnn_output_option = settings.rnn_output_option

        bidirectional_factor = 2 if self.bidirectional is True else 1
        last_input_size = (
            self.hidden_size * bidirectional_factor
            if self.rnn_output_option == "mean"
            else self.hidden_size * bidirectional_factor * self.num_layers
        )

        # Define layers
        self.rnn_layer = getattr(torch.nn, self.layer_type.upper())(
            input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=0,  # self.dropout,
            bidirectional=self.bidirectional,
        )
        # self.output_dropout_layer = torch.nn.Dropout(self.dropout)
        self.output_class_layer = torch.nn.Linear(last_input_size, self.output_size)
        # regression does not use mean vs standard outputs
        self.output_peak_layer = torch.nn.Linear(
            self.hidden_size * bidirectional_factor, 1
        )

    def forward(self, x, mean_field_inference=False):
        # Reminder
        # out = packed output from last layer
        # out has dim (seq_len, batch_size, hidden_size) when unpacked
        # hidden = (hn, cn) for lstm (only final h from each pass and layer)
        # hidden = hn for GRU and RNN (only final h from each pass and layer)
        # hn has dim (num_layers * num_directions, batch, hidden_size)
        # cn has dim (num_layers * num_directions, batch, hidden_size)
        # assuming num_directions = 1, num_layers = 2 :
        # hn[-1, -1] == out[len, -1] where len is the len of the seq at batch index == -1

        x, hidden = self.rnn_layer(x)

        # Output options
        # Standard: all layers, only end of pass
        #    - take last pass in all layers (hidden)
        #    - reshape and apply dropout
        #    - use h20 to obtain output (h2o input: hidden_size*num_layers*bi)
        # Mean: last layer, mean on sequence
        #    - take packed output from last layer (out) that contains all time steps for the last layer
        #    - find where padding was done and create a mask for those values, apply this mask
        #    - take a mean for the whole sequence (time_steps)
        #    - use h2o to obtain output (beware! it is only one layer deep since it is the last one only)

        # Classification
        if self.rnn_output_option == "standard":
            # Special case for lstm where hidden = (h, c)
            if self.layer_type == "lstm":
                hn = hidden[0]
            else:
                hn = hidden
            # hn is (num_layers * num_directions, batch, hidden_size)
            hn = hn.permute(1, 2, 0).contiguous()
            # hn now is (batch, hidden size, num_layers * num_directions)
            batch_size = hn.shape[0]
            x_class = hn.view(batch_size, -1)
            # x_class is (batch, hidden size * num_layers * num_directions)

        if self.rnn_output_option == "mean":
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x_class, lens = torch.nn.utils.rnn.pad_packed_sequence(x)
                # x_class is (seq_len, batch, hidden size * num_directions)

                # take mean over seq_len
                x_class = x_class.sum(0) / lens.unsqueeze(-1).float().to(x_class.device)
                # x_class is (batch, hidden_size * num_directions)
            else:
                x_class = x.mean(0)

        # Peak prediction
        # for each time step, we predict a peak light distance
        # it doesnt make sense to do in this case mean pooling or just taking the last hidden state
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x_unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(x)
            outpeak = self.output_peak_layer(x_unpacked)
            # now I need to mask padded values
            mask = torch.arange(lens.max().item()).view(1, -1).to(outpeak.device)
            lens = lens.view(-1, 1).float().to(outpeak.device)
            # lens == (B, 1)
            # torch.arange == (1, max_len)
            # mask (B, max_len)
            # if array 1D = 0 1 2 3
            # max_len = 4
            maskpeak = (mask.float() < lens).float()
            # reshape mask to match outpeak
            maskpeak = maskpeak.transpose(1, 0).contiguous()
        else:
            outpeak = self.output_peak_layer(x)
            maskpeak = None

        # apply dropout
        # x_class = self.output_dropout_layer(x_class)
        # Final projection layer
        outclass = self.output_class_layer(x_class)

        return outclass, outpeak, maskpeak
