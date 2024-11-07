// SPDX-License-Identifier: Bhide License
pragma solidity ^0.8.0;

contract Bank {
    mapping(address => uint256) private balances;

    // Create account (initializes balance to 0, but it's not strictly necessary)
    function createAccount() public {
        balances[msg.sender] = 0;
    }

    // Deposit money into the account (payable to receive Ether)
    function deposit() public payable {
        require(msg.value > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += msg.value;
    }

    // Withdraw money from the account
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        payable(msg.sender).transfer(amount); // Sending Ether to the user
        balances[msg.sender] -= amount;
    }

    // Transfer money to another account
    function transfer(address recipient, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
    }

    // Get the balance of the account
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
