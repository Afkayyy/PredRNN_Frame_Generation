import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

# Define STM Cell (modified from PredRNN)
class STMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(STMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_channels + hidden_channels * 2, 
            hidden_channels * 4, 
            kernel_size, 
            padding=padding
        )

    def forward(self, x, h, c, m):
        combined = torch.cat([x, h, m], dim=1)  # Combine input, hidden state, and memory
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        m_next = m + c_next  # Update spatio-temporal memory
        return h_next, c_next, m_next

# Define PredRNN Model with STM Cells
class PredRNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels):
        super(PredRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.cells = nn.ModuleList([
            STMCell(
                input_channels if i == 0 else hidden_channels, 
                hidden_channels, 
                kernel_size
            ) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x, predict_steps=1):
        batch_size, seq_len, _, height, width = x.size()
        h, c, m = self.init_hidden(batch_size, height, width, x.device)

        # Process input sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i], m[i] = cell(x_t, h[i], c[i], m[i])
                x_t = h[i]

        # Generate future frames
        outputs = []
        x_t = self.conv_out(h[-1])  # First predicted frame based on last hidden state
        outputs.append(x_t)

        for _ in range(predict_steps - 1):
            for i, cell in enumerate(self.cells):
                h[i], c[i], m[i] = cell(x_t, h[i], c[i], m[i])
                x_t = h[i]
            x_t = self.conv_out(h[-1])
            outputs.append(x_t)

        return torch.stack(outputs, dim=1)

    def init_hidden(self, batch_size, height, width, device):
        h = [torch.zeros(batch_size, self.hidden_channels, height, width).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_channels, height, width).to(device) for _ in range(self.num_layers)]
        m = [torch.zeros(batch_size, self.hidden_channels, height, width).to(device) for _ in range(self.num_layers)]
        return h, c, m

def load_predrnn_model(device):
    model = PredRNN(input_channels=1, hidden_channels=64, kernel_size=3, num_layers=2, output_channels=1)
    state_dict_path = "models/predrnn_model.pth"  # Update the path to the PreDNN model weights
    state_dict = torch.load(state_dict_path, map_location=device)

    # Adjust for potential 'module.' prefix in keys
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    
    # Load only matching parameters and ignore others
    model_state_dict = model.state_dict()
    for name, param in new_state_dict.items():
        if name in model_state_dict and param.size() == model_state_dict[name].size():
            model_state_dict[name].copy_(param)

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model


def predict_PredRNN(input_tensor, device, predict_steps=5):
    model = load_predrnn_model(device)
    with torch.no_grad():
        predicted_frames = model(input_tensor, predict_steps=predict_steps)
    return predicted_frames.squeeze(0).cpu().numpy()  # Shape: (predict_steps, 1, height, width)
