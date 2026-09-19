// ============================================================================
// SRAM-R (Static Rotor Auxiliary Register File) RTL
// ============================================================================

module sram_r_register #(
    parameter DATA_WIDTH = 128, // 4 x 32-bit float rotor components (scalar, e12, e23, e31)
    parameter ADDR_WIDTH = 10,  // 1024 tag entries
    parameter DEPTH      = 1024
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  we,          // Write enable (Pinning)
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] wdata_delta, // Delta Rotor input
    output reg  [DATA_WIDTH-1:0] rdata_delta, // Restored Delta Rotor output
    output reg                   hit
);

    reg [DATA_WIDTH-1:0] sram_array [0:DEPTH-1];
    reg [DEPTH-1:0]      valid_array;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_array <= {DEPTH{1'b0}};
            rdata_delta <= {DATA_WIDTH{1'b0}};
            hit         <= 1'b0;
            for (i = 0; i < DEPTH; i = i + 1) begin
                sram_array[i] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            if (we) begin
                sram_array[addr]  <= wdata_delta;
                valid_array[addr] <= 1'b1;
            end

            rdata_delta <= sram_array[addr];
            hit         <= valid_array[addr];
        end
    end

endmodule
