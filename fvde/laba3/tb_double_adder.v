// Testbench for IEEE Double Precision Floating Point Adder

`timescale 1ns/1ps

module tb_double_adder;

  // Inputs to the DUT
  reg clk;
  reg rst;
  reg [63:0] input_a;
  reg [63:0] input_b;
  reg input_a_stb;
  reg input_b_stb;
  reg output_z_ack;
  reg stb;
  // Outputs from the DUT
  wire [63:0] output_z;
  wire output_z_stb;
  wire input_a_ack;
  wire input_b_ack;
  reg  start;
  // Instantiate the DUT
  double_adder dut (
    .clk(clk),
    .rst(rst),
    .input_a(input_a),
    .input_b(input_b),
    .input_a_stb(stb),
    .input_b_stb(stb),
    .output_z_ack(output_z_ack),
    .output_z(output_z),
    .output_z_stb(output_z_stb),
    .input_a_ack(input_a_ack),
    .input_b_ack(input_b_ack)
  );
  // assign output_z_ack = 0;
  // Test variables
  reg [63:0] random_operand_a;
  reg [63:0] random_operand_b;
  // Test procedure
  initial begin
    // Initialize inputs
    rst <= 1;
    start<= 1;
    stb <= 0;
    clk <= 0;
    input_a_stb <= 0;
    input_b_stb <= 0;
    #50 rst <= 0;
    end
  // Clock generation
  initial
  begin
    #1000000 $finish;
  end
  initial begin
    while (1) begin
     #5 clk = ~clk; // 100 MHz clock
    end
  end

  always @(posedge clk)
  begin
      random_operand_a = {$random, $random};
      random_operand_b = {$random, $random}; 
      input_a = random_operand_a;
      input_b = random_operand_b;
      // Wait for DUT result
      stb <= $random%2;
       if (output_z_stb) begin
            output_z_ack <= 1'b1;

          end
        else output_z_ack <= 1'b0;

      // Display results
      if (output_z_stb) begin
        $display("Operand A: %h, Operand B: %h", input_a, input_b);
        $display("DUT Result: %h", output_z);
      end
  end

endmodule