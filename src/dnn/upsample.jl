function upsample_kernel(state, y, x, height, width, channels, batch, stride, scale)
    i = @linearidx y state
    
    y_idx = i
    y_h = (i - 1) % (height * stride[1]) + 1
    i = (i - 1) ÷ (height * stride[1])
    y_w = i % (width * stride[2]) + 1
    i = i ÷ (width * stride[2])
    y_c = i % channels + 1
    i = i ÷ channels
    y_b = i % batch + 1
    
    x_h = (y_h - 1) ÷ stride[1] + 1
    x_w = (y_w - 1) ÷ stride[2] + 1
    x_c = y_c
    x_idx = (y_b - 1) * width * height * channels + (x_c - 1) * width * height + (x_w - 1) * height + x_h
    
    @inbounds y[y_idx] = scale * x[x_idx]
    
    return nothing
end

function upsample(x::CuArray, stride, scale = 1)
    (height, width, channels, batch) = size(x)
    y = similar(x, (height * stride[1], width * stride[2], channels, batch))
    gpu_call(upsample_kernel, y, (y, x, height, width, channels, batch, stride, scale))
    return y
end

function ∇upsample_kernel(state, y, x, height, width, channels, batch, stride, scale)
    i = @linearidx y state

    y_idx = i
    y_h = (i - 1) % (height * stride[1]) + 1
    i = (i - 1) ÷ (height * stride[1])
    y_w = i % (width * stride[2]) + 1
    i = i ÷ (width * stride[2])
    y_c = i % channels + 1
    i = i ÷ channels
    y_b = i % batch + 1

    x_h = (y_h - 1) ÷ stride[1] + 1
    x_w = (y_w - 1) ÷ stride[2] + 1
    x_c = y_c
    x_idx = (y_b - 1) * width * height * channels + (x_c - 1) * width * height + (x_w - 1) * height + x_h

    @inbounds x[x_idx] += y[y_idx] / scale

    return nothing
end

function ∇upsample(dy::CuArray, stride, scale = 1)
    (height, width, channels, batch) = size(dy)
    @assert height % stride[1] == 0
    @assert width % stride[2] == 0
    dx = similar(dy, (height ÷ stride[1], width ÷ stride[2], channels, batch))
    gpu_call(∇upsample_kernel, dy, (dy, dx, height, width, channels, batch, stride, scale))
    return dx
end
