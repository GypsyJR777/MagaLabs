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
  logic last_input_a;
  logic last_input_b;

  // Outputs from the DUT
  logic [63:0] output_z;
  logic output_z_stb;
  logic input_a_ack;
  logic input_b_ack;

  // Variables
  logic [63:0] random_operand_a;
  logic [63:0] random_operand_b;
  bit [63:0] reference_result; // Declare reference_result for the DPI function
  int iteration_count = 0;       // Iteration counter
  real operand_a_real, operand_b_real, result_real;

  // Instantiate the DUT
  double_adder dut (
    .clk(clk),
    .rst(rst),
    .input_a(input_a),
    .input_b(input_b),
    .input_a_stb(input_a_stb),
    .input_b_stb(input_b_stb),
    .output_z_ack(output_z_ack),
    .output_z(output_z),
    .output_z_stb(output_z_stb),
    .input_a_ack(input_a_ack),
    .input_b_ack(input_b_ack)
  );

  import "DPI-C" function void reference_adder(
    input longint a,
    input longint b,
    output longint result
  );

  initial begin
    clk  <= 1'b0;
    rst  <= 1'b1;
    #50 rst <= 1'b0;
    iteration_count <= 0;
  end
    
  initial begin
    #1000000 $finish;
  end

  always #5 clk = ~clk;

  initial begin
    input_a_stb = 0;
    input_b_stb = 0;

    output_z_ack = 0;
    input_a = 64'b0;
    input_b = 64'b0;
  end

always @(posedge clk)
  begin
        if (rst) begin
            input_a_stb <= 0;
            input_b_stb <= 0;
            output_z_ack <= 0;
            input_a <= 64'b0;
            input_b <= 64'b0;
        end else begin
            if (output_z_stb && !output_z_ack) begin
                iteration_count <= iteration_count + 1;
                output_z_ack <= 1'b1;
                $display("Operand A: %h, Operand B: %h", last_input_a, last_input_b);
                $display("DUT Result: %h, C++ result: %h", output_z, reference_result);
                operand_a_real = $bitstoreal({32'b0, last_input_a}); // Для корректности double (64 бит)
                operand_b_real = $bitstoreal({32'b0, last_input_b});
                result_real = $bitstoreal({32'b0, output_z});
                $display("REAL: Operand A: %e, Operand B: %e, Result DUT: %e, Result C++: %e", 
                  operand_a_real, operand_b_real, result_real, $bitstoreal({32'b0, reference_result})
                );
                if (output_z != reference_result) begin
                  $display("DUT res and C++ not matching");
                  $finish();
                end
            end else if (!output_z_stb) begin
                output_z_ack <= 1'b0;
            end;
        end
    
        if (!input_a_stb && !input_b_stb && !output_z_ack) begin
                input_a_stb <= 1'b1;
                input_b_stb <= 1'b1;
                input_a = {$random(), $random()};
                input_b = {$random(), $random()};
                reference_adder(input_a, input_b, reference_result);
                last_input_a <= input_a;
                last_input_b <= input_b;
        end else if (input_a_ack && input_b_ack) begin
                input_a_stb <= 1'b0; 
                input_b_stb <= 1'b0; 
        end


      // random_operand_a = {$random, $random};
      // random_operand_b = {$random, $random}; 
      // input_a = random_operand_a;
      // input_b = random_operand_b;
      // // Wait for DUT result
      // stb <= $random%2;
      //  if (output_z_stb) begin
      //       output_z_ack <= 1'b1;

      //     end
      //   else output_z_ack <= 1'b0;

      // // Display results
      // if (output_z_stb) begin
      //   $display("Operand A: %h, Operand B: %h", input_a, input_b);
      //   $display("DUT Result: %h", output_z);
      // end
  end

endmodule