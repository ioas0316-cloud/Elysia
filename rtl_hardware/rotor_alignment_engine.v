// ============================================================================
// Rotor Alignment Engine (RAE) RTL
// ============================================================================

module rotor_alignment_engine #(
    parameter WIDTH = 32
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid_in,
    // Global Rotor components
    input  wire [WIDTH-1:0] r_glob_s,
    input  wire [WIDTH-1:0] r_glob_xy,
    input  wire [WIDTH-1:0] r_glob_yz,
    input  wire [WIDTH-1:0] r_glob_zx,
    // Delta Rotor components
    input  wire [WIDTH-1:0] r_delta_s,
    input  wire [WIDTH-1:0] r_delta_xy,
    input  wire [WIDTH-1:0] r_delta_yz,
    input  wire [WIDTH-1:0] r_delta_zx,
    // Restored Active Rotor components
    output reg  [WIDTH-1:0] r_act_s,
    output reg  [WIDTH-1:0] r_act_xy,
    output reg  [WIDTH-1:0] r_act_yz,
    output reg  [WIDTH-1:0] r_act_zx,
    output reg              valid_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_act_s   <= {WIDTH{1'b0}};
            r_act_xy  <= {WIDTH{1'b0}};
            r_act_yz  <= {WIDTH{1'b0}};
            r_act_zx  <= {WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // Simplified Geometric Product calculation for RTL
            r_act_s   <= (r_glob_s  * r_delta_s)  - (r_glob_xy * r_delta_xy) - (r_glob_yz * r_delta_yz) - (r_glob_zx * r_delta_zx);
            r_act_xy  <= (r_glob_s  * r_delta_xy) + (r_glob_xy * r_delta_s)  - (r_glob_yz * r_delta_zx) + (r_glob_zx * r_delta_yz);
            r_act_yz  <= (r_glob_s  * r_delta_yz) + (r_glob_yz * r_delta_s)  - (r_glob_zx * r_delta_xy) + (r_glob_xy * r_delta_zx);
            r_act_zx  <= (r_glob_s  * r_delta_zx) + (r_glob_zx * r_delta_s)  - (r_glob_xy * r_delta_yz) + (r_glob_yz * r_delta_xy);
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
