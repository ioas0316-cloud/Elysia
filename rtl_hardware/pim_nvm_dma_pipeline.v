// ============================================================================
// PIM / NVM Async DMA Inscription Pipeline RTL
// ============================================================================

module pim_nvm_dma_pipeline #(
    parameter DATA_WIDTH = 288, // 3x3 Metric Tensor = 9 x 32-bit floats
    parameter ADDR_WIDTH = 32
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  dma_trigger,
    input  wire [ADDR_WIDTH-1:0] ssd_target_addr,
    input  wire [DATA_WIDTH-1:0] g_mem_packet,
    output reg                   dma_busy,
    output reg                   dma_done,
    output reg  [ADDR_WIDTH-1:0] bus_addr,
    output reg  [DATA_WIDTH-1:0] bus_data,
    output reg                   bus_write_en
);

    localparam IDLE  = 2'b00;
    localparam WRITE = 2'b01;
    localparam DONE  = 2'b10;

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= IDLE;
            dma_busy     <= 1'b0;
            dma_done     <= 1'b0;
            bus_addr     <= {ADDR_WIDTH{1'b0}};
            bus_data     <= {DATA_WIDTH{1'b0}};
            bus_write_en <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    dma_done <= 1'b0;
                    if (dma_trigger) begin
                        state        <= WRITE;
                        dma_busy     <= 1'b1;
                        bus_addr     <= ssd_target_addr;
                        bus_data     <= g_mem_packet;
                        bus_write_en <= 1'b1;
                    end
                end

                WRITE: begin
                    bus_write_en <= 1'b0;
                    dma_busy     <= 1'b0;
                    dma_done     <= 1'b1;
                    state        <= DONE;
                end

                DONE: begin
                    dma_done <= 1'b0;
                    state    <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
