// Testbench for IEEE Double Precision Floating Point Adder

`timescale 1ns/1ps

module tb_double_adder;

  // Inputs to the DUT
  logic clk;
  logic rst;
  logic [63:0] input_a;
  logic [63:0] input_b;
  logic input_a_stb;
  logic input_b_stb;
  logic output_z_ack;
  logic stb;

  // Outputs from the DUT
  logic [63:0] output_z;
  logic output_z_stb;
  logic input_a_ack;
  logic input_b_ack;

  // Variables
  logic [63:0] random_operand_a;
  logic [63:0] random_operand_b;
  logic [63:0] reference_result; // Declare reference_result for the DPI function
  int iteration_count = 0;       // Iteration counter

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

  // Import DPI function
  import "DPI-C" function void reference_adder(
    input longint a,
    input longint b,
    output longint result
  );

  // Test procedure
  initial begin
    // Initialize inputs
    rst = 1;
    clk = 0;
    stb = 0;
    // output_z_ack = 0;

    // Release reset after some time
    #50 rst = 0;

    // Run the test
     while (iteration_count < 13)  begin
      #10;
      // Generate random operands
      random_operand_a = {$random, $random};
      random_operand_b = {$random, $random};

      // Assign to inputs
      input_a = random_operand_a;
      input_b = random_operand_b;

      // Wait for DUT result
      stb <= $random%2;
       if (output_z_stb) begin
            output_z_ack <= 1'b1;

          end
        else output_z_ack <= 1'b0;

      // Display results
      $display("Operand A: %h, Operand B: %h", random_operand_a, random_operand_b);

      // Call the reference function
      reference_adder(random_operand_a, random_operand_b, reference_result);

      // Display results
      // $display("Operand A: %h, Operand B: %h", random_operand_a, random_operand_b);
      $display("DUT Result: %h, Reference Result: %h", output_z, reference_result);

      // Check for mismatches
      if (output_z != reference_result) begin
        $display("ERROR: Mismatch detected!");
      end else begin
        $display("Match: Results are correct.");
      end

      // Wait before the next iteration
      #20;
      iteration_count += 1;
    end
  end

  // Clock generation



endmodule
